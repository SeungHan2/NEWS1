import os
import time
import json
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------------------
# 환경 변수 및 설정
# ----------------------------------------
load_dotenv()

def get_openai_api_key() -> str:
    """
    OPENAI_API_KEY 환경변수를 읽어서 공백 제거 후 리턴.
    (로컬 .env / GitHub Actions env 둘 다 여기로 들어옴)
    """
    key = os.getenv("OPENAI_API_KEY", "")
    return key.strip()

OPENAI_API_KEY = get_openai_api_key()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not OPENAI_API_KEY:
    # 여기서 바로 죽여버리면, GitHub Actions 로그에서 원인을 바로 알 수 있음
    raise SystemExit(
        "[ERROR] OPENAI_API_KEY 환경변수가 비어 있습니다.\n"
        " - 로컬: .env 파일에 OPENAI_API_KEY=... 추가\n"
        " - GitHub Actions: workflow yml에서 env: OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }} 로 전달 필요"
    )

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 사용할 GPT 모델 (원하면 환경변수로 빼도 됨)
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-4.1-mini").strip()


PRESS_LIST: List[Tuple[str, str]] = [
    ("동아일보", "020"),
    ("한국일보", "469"),
    ("조선일보", "023"),
    ("중앙일보", "025"),
    ("한겨레", "028"),
    ("경향신문", "032"),
]

# ----------------------------------------
# [Part 1] 네이버 1면 링크 수집
# ----------------------------------------
def get_kst_today() -> str:
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc + timedelta(hours=9)
    return now_kst.strftime("%Y%m%d")

def fetch_html(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url.strip(), headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

def extract_a1_links(html: str, page_url: str, press_code: str, date: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if f"/article/newspaper/{press_code}/" not in href:
            continue
        if f"date={date}" not in href:
            continue
        full_url = urljoin(page_url, href)

        is_a1 = False
        parent = a
        for _ in range(6):
            parent = parent.parent
            if parent is None:
                break
            text = parent.get_text(" ", strip=True)
            if any(key in text for key in ["A1면", "A01면", "1면", "1 面"]):
                is_a1 = True
                break
        if is_a1:
            candidates.append(full_url)

    if not candidates:  # Fallback
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/article/newspaper/{press_code}/" in href and f"date={date}" in href:
                full_url = urljoin(page_url, href)
                if full_url not in seen:
                    candidates.append(full_url)
                    seen.add(full_url)
            if len(candidates) >= 4:
                break
    return list(set(candidates))

def collect_naver_news_links() -> List[Dict[str, str]]:
    date = get_kst_today()
    print(f"[INFO] {date}일자 1면 기사 수집 시작")
    all_items = []
    for press_name, press_code in PRESS_LIST:
        url = ""
        try:
            url = f"https://media.naver.com/press/{press_code}/newspaper?date={date}".strip()
            html = fetch_html(url)
            links = extract_a1_links(html, url, press_code, date)
            for link in links:
                all_items.append({"source": press_name, "url": link})
        except Exception as e:
            print(f"  [에러] {press_name} 수집 실패: {e}")
            print(f"  [URL] 요청 실패 URL: {url}")
    return all_items

# ----------------------------------------
# [Part 2] 본문 크롤링
# ----------------------------------------
def fetch_single_article_content(item: dict) -> dict:
    try:
        resp = requests.get(item["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        selectors = ["div#dic_area", "div#newsEndContents", "div.newsct_article", "div#articleBodyContents"]
        content = ""
        for selector in selectors:
            node = soup.select_one(selector)
            if node:
                content = node.get_text("\n", strip=True)
                break
        return {
            "source": item["source"],
            "url": item["url"],
            "content": content[:4000] if content else "본문 없음"
        }
    except Exception:
        return item

def fetch_contents_parallel(items: list) -> list:
    print(f"[INFO] 총 {len(items)}개 기사 본문 크롤링 중...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_single_article_content, items))
    return results

# ----------------------------------------
# [Part 3] GPT 분석 (리포트 작성)
# ----------------------------------------
def analyze_with_gpt(articles: list) -> dict:
    if client is None:
        print("[CRITICAL ERROR] OpenAI 클라이언트가 초기화되지 않았습니다. OPENAI_API_KEY를 확인하세요.")
        return {"topics": []}

    print(f"[INFO] {GPT_MODEL_NAME} 분석 요청 시작...")

    articles_text = ""
    for i, art in enumerate(articles):
        articles_text += f"[ID:{i}] 언론사:{art['source']} | 내용:{art['content'][:2000]}\n"

    # 프롬프트에서 JSON 형식 강하게 요구
    prompt = f"""
    너는 전문 뉴스 에디터다. 오늘자 신문 1면 기사들을 종합하여 고품질 리포트를 작성하라.

    [요구사항]
    1. 기사들을 유사한 주제(정치, 경제, 사회 등)로 그룹화하라.
    2. **주제별 통합 기사 작성**: 각 주제에 대해 개별 기사를 단순히 나열하지 말고, 모든 내용을 종합하여 **하나의 완결된 심층 기사**로 새로 써라.
        - **분량**: 반드시 **최소 500자 이상**의 상세한 글로 작성할 것.
        - **구성**: 기사의 배경, 현재 상황, 언론사별 주요 주장, 그리고 향후 전망이나 전문가 분석 등 다각도의 관점을 포함하여 작성할 것.
        - **톤**: 전문가가 작성한 객관적인 논조의 기사 형태를 유지할 것.
    3. **요약본(Bullets)**: 바쁜 독자를 위해, 통합 기사의 내용을 3줄 이내의 핵심 단문(Bullet point)으로 요약하라.
    4. 아래 JSON 스키마를 **반드시 그대로 따르는 유효한 JSON 문자열만** 출력하라.
       - JSON 밖의 다른 텍스트(설명, 마크다운, 코드블록 등)는 절대 출력하지 마라.

    [JSON 구조]
    {{
        "topics": [
            {{
                "title": "주제 제목 (예: 금투세 폐지 논란 가열)",
                "ids": [0, 2, 5],
                "summary_bullets": ["핵심 내용 1", "핵심 내용 2"],
                "full_article": "여기에 GPT가 새로 작성한 통합 기사 전문(줄글로 작성). 500자 이상을 채우도록 노력해야 한다."
            }}
        ]
    }}

    [기사 데이터]
    {articles_text}
    """

    response = None
    try:
        # 🔴 여기에서 더 이상 response_format 인자를 사용하지 않는다
        response = client.responses.create(
            model=GPT_MODEL_NAME,
            input=prompt,
        )

        # OpenAI responses 구조에서 텍스트 추출
        raw_text = ""
        try:
            raw_text = response.output[0].content[0].text.strip()
        except Exception as e:
            print(f"[WARN] response.output에서 텍스트 추출 실패, fallback 시도: {e}")
            if hasattr(response, "output_text"):
                raw_text = response.output_text.strip()
            else:
                raw_text = str(response).strip()

        # 혹시라도 ```json ``` 등 코드블록으로 감싸져 있으면 제거
        if raw_text.startswith("```json"):
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.removeprefix("```").removesuffix("```").strip()

        return json.loads(raw_text)

    except json.JSONDecodeError as e:
        print(f"[CRITICAL ERROR] JSON 디코딩 실패: {e}")
        print("--- GPT Raw Output Start ---")
        if response is not None:
            try:
                print(response.output[0].content[0].text)
            except Exception:
                print(str(response))
        else:
            print("No response object available.")
        print("--- GPT Raw Output End ---")
        return {"topics": []}

    except Exception as e:
        print(f"[CRITICAL ERROR] GPT 분석 중 기타 에러 발생: {e}")
        return {"topics": []}


# ----------------------------------------
# [Part 4] Telegraph 페이지 생성 (웹뷰)
# ----------------------------------------
def create_telegraph_simple(title: str, text_body: str) -> str:
    """간단한 텍스트 기반 Telegraph 페이지 생성"""
    try:
        telegraph_account_url = "https://api.telegra.ph/createAccount?short_name=NewsAI"
        print(f"[DEBUG] Telegraph Account URL: {telegraph_account_url}")

        r = requests.get(telegraph_account_url).json()
        token = r["result"]["access_token"]

        content_nodes = []
        content_nodes.append({"tag": "h3", "children": ["AI 통합 리포트"]})

        current_p_children = []
        for line in text_body.split("\n"):
            line = line.strip()
            if not line and current_p_children:
                content_nodes.append({"tag": "p", "children": current_p_children})
                current_p_children = []
            elif line:
                current_p_children.append(line)

        if current_p_children:
            content_nodes.append({"tag": "p", "children": current_p_children})

        data = {
            "access_token": token,
            "title": title,
            "content": json.dumps(content_nodes),
            "return_content": False,
        }

        telegraph_create_page_url = "https://api.telegra.ph/createPage"
        resp = requests.post(telegraph_create_page_url, data=data).json()

        if resp.get("ok"):
            return resp["result"]["url"]
        else:
            print(f"Telegraph API 오류: {resp.get('error')}")
            return ""
    except Exception as e:
        print(f"Telegraph 생성 실패: {e}")
        return ""

# ----------------------------------------
# [Part 5] 텔레그램 전송 (HTML 모드)
# ----------------------------------------
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARNING] 텔레그램 토큰 또는 채팅 ID가 없어 전송을 건너킵니다.")
        return

    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"

    masked_url = url.replace(TELEGRAM_BOT_TOKEN, "***masked***")
    print(f"[DEBUG] Telegram URL length: {len(url)}")
    print(f"[DEBUG] Telegram URL fragment (masked): {masked_url[:70]}")

    chunk_size = 4000
    for i in range(0, len(message), chunk_size):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message[i : i + chunk_size],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        requests.post(url, data=payload)
        time.sleep(0.5)

# ----------------------------------------
# 메인 실행
# ----------------------------------------
def main():
    # 1. 링크 수집 및 통계
    links = collect_naver_news_links()
    if not links:
        print("수집된 기사가 없어 종료합니다.")
        return

    stats = {}
    for item in links:
        stats[item["source"]] = stats.get(item["source"], 0) + 1

    header_stats = " | ".join([f"{k} {v}" for k, v in stats.items()])

    # 2. 본문 크롤링
    contents = fetch_contents_parallel(links)

    # 3. GPT 분석
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY가 없어 분석을 생략합니다.")
        return

    result = analyze_with_gpt(contents)

    # 4. 리포트 및 웹뷰 컨텐츠 생성
    today_str = get_kst_today()

    telegram_msg = f"<b>🗞 {today_str} 신문 1면 브리핑</b>\n\n"
    telegram_msg += f"📊 <b>수집 현황:</b> {header_stats}\n\n"

    webview_text = f"📰 {today_str} 신문 1면 통합 리포트\n\n[수집 현황] {header_stats}\n\n"

    topics = result.get("topics", [])

    # 주제별 기사 수 내림차순 정렬
    topics.sort(key=lambda t: len(t.get("ids", [])), reverse=True)

    if not topics:
        telegram_msg += "<b>⚠️ 리포트 생성 실패: 분석 과정에서 오류가 발생했거나, AI가 답변을 거부했습니다. GitHub Actions 로그를 확인하세요.</b>"
        webview_text = "리포트 생성 실패"
    else:
        for topic in topics:
            title = topic.get("title", "무제")
            ids = topic.get("ids", [])
            bullets = topic.get("summary_bullets", [])
            full_article = topic.get("full_article", "")

            # --- 텔레그램 메시지 구성 ---
            telegram_msg += f"━━━━━━━━━━━━━━\n"
            telegram_msg += f"📌 <b>{title}</b> ({len(ids)}건)\n"

            link_tags = []
            for idx in ids:
                if idx < len(contents):
                    item = contents[idx]
                    link_tags.append(f"<a href='{item['url']}'>{item['source']}</a>")
            telegram_msg += f"🔗 {' , '.join(link_tags)}\n\n"

            for bullet in bullets:
                telegram_msg += f"• {bullet}\n"
            telegram_msg += "\n"

            # --- 웹뷰 텍스트 구성 ---
            webview_text += f"\n### 📌 {title} ({len(ids)}건)\n"
            webview_text += "\n[핵심 요약]\n"
            for bullet in bullets:
                webview_text += f" - {bullet}\n"
            webview_text += "\n[통합 심층 기사]\n"
            webview_text += f"{full_article}\n"
            webview_text += "\n\n"

    # 5. Telegraph 페이지 생성 (긴 화면용)
    webview_url = create_telegraph_simple(f"{today_str} 조간 브리핑", webview_text)

    if webview_url:
        telegram_msg += f"\n\n📱 <b><a href='{webview_url}'>👉 전체 리포트 크게 보기 (Safari/Web)</a></b>"

    # 6. 전송
    print("[INFO] 텔레그램 전송 중...")
    send_telegram(telegram_msg)
    print("[INFO] 완료.")

if __name__ == "__main__":
    main()
