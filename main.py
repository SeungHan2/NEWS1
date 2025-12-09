import os
import time
import json
import html
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# [NEW] Google Generative AI 라이브러리 임포트
import google.generativeai as genai
from google.api_core import retry

# ----------------------------------------
# 환경 변수 및 설정
# ----------------------------------------
load_dotenv()

def get_gemini_api_key() -> str:
    """
    GEMINI_API_KEY 환경변수를 읽어서 공백 제거 후 리턴.
    """
    key = os.getenv("GEMINI_API_KEY", "")
    return key.strip()

GEMINI_API_KEY = get_gemini_api_key()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not GEMINI_API_KEY:
    raise SystemExit(
        "[ERROR] GEMINI_API_KEY 환경변수가 비어 있습니다.\n"
        " - .env 파일에 GEMINI_API_KEY=... 를 추가하세요.\n"
        " - Google AI Studio(https://aistudio.google.com/)에서 키를 발급받을 수 있습니다."
    )

# [NEW] Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)

# 사용할 모델 (기본값: gemini-1.5-flash)
# 뉴스 요약용으로는 1.5 Flash가 속도/비용 면에서 유리하며,
# 더 깊은 추론이 필요하면 'gemini-1.5-pro'로 변경하세요.
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash").strip()

def escape_html(text: str) -> str:
    """Escape user/content strings for safe Telegram HTML."""
    return html.escape(text or "", quote=True)

PRESS_LIST: List[Tuple[str, str]] = [
    ("동아일보", "020"),
    ("한국일보", "469"),
    ("조선일보", "023"),
    ("중앙일보", "025"),
    ("한겨레", "028"),
    ("경향신문", "032"),
]

# ----------------------------------------
# [Part 1] 네이버 1면 링크 수집 (기존 동일)
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
    return all_items

# ----------------------------------------
# [Part 2] 본문 크롤링 (기존 동일)
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
# [Part 3] Gemini 분석 (리포트 작성) - 변경됨
# ----------------------------------------
def analyze_with_gemini(articles: list) -> dict:
    print(f"[INFO] {GEMINI_MODEL_NAME} 분석 요청 시작...")

    # 기사 본문 모으기
    articles_text = ""
    for i, art in enumerate(articles):
        articles_text += f"[ID:{i}] 언론사:{art['source']} | 내용:{art['content'][:2000]}\n"

    # Gemini에게 요청할 시스템 프롬프트
    system_instruction = """
    너는 전문 뉴스 에디터다. 오늘자 신문 1면 기사들을 종합하여 고품질 리포트를 작성하라.
    
    [요구사항]
    1. 기사들을 유사한 주제(정치, 경제, 사회 등)로 그룹화하라.
    2. 주제별 통합 기사 작성: 각 주제에 대해 개별 기사를 단순히 나열하지 말고, 모든 내용을 종합하여 하나의 완결된 심층 기사로 새로 써라.
       - 분량: 최소 500자 이상.
       - 구성: 배경, 현황, 언론사별 주요 주장, 전망 등을 포함.
       - 톤: 객관적인 논조 유지.
    3. 요약본(Bullets): 바쁜 독자를 위해 3줄 이내 핵심 요약.
    4. 언론사별 비판/논조 정리: 해당 주제 내 기사들의 언론사별 논조(비판, 옹호, 우려 등)를 요약.
    
    반드시 아래의 JSON 스키마를 준수하여 출력해야 한다.
    """

    # Gemini 1.5부터는 JSON 스키마를 명시적으로 제어할 수 있으나, 
    # 여기서는 프롬프트 내 예시와 response_mime_type 설정을 통해 제어합니다.
    prompt = f"""
    [기사 데이터]
    {articles_text}

    [출력 JSON 형식을 엄수할 것]
    {{
        "topics": [
            {{
                "title": "주제 제목",
                "ids": [0, 2],
                "summary_bullets": ["요약1", "요약2"],
                "full_article": "통합 줄글 기사 (500자 이상)",
                "press_critiques": [
                    {{
                        "source": "언론사명",
                        "position": "논조 및 주장 요약",
                        "tone": "비판적/옹호적/중립적"
                    }}
                ]
            }}
        ]
    }}
    """

    try:
        # 모델 설정 (JSON 모드 활성화)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            system_instruction=system_instruction,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.3, # 뉴스 분석이므로 창의성보다는 정확성 중요
            }
        )
        
        # API 요청 (Retry 정책 적용 권장)
        response = model.generate_content(prompt, request_options={"retry": retry.Retry(predicate=retry.if_transient_error)})
        
        # 결과 텍스트 추출 및 JSON 파싱
        raw_text = response.text
        return json.loads(raw_text)

    except json.JSONDecodeError as e:
        print(f"[CRITICAL ERROR] JSON 디코딩 실패: {e}")
        # 디버깅용 출력
        # print(raw_text) 
        return {"topics": []}

    except Exception as e:
        print(f"[CRITICAL ERROR] Gemini 분석 중 에러 발생: {e}")
        return {"topics": []}


# ----------------------------------------
# [Part 4] Telegraph 페이지 생성 (기존 동일)
# ----------------------------------------
def create_telegraph_simple(title: str, text_body: str) -> str:
    try:
        telegraph_account_url = "https://api.telegra.ph/createAccount?short_name=NewsAI"
        r = requests.get(telegraph_account_url, timeout=10).json()
        token = r["result"]["access_token"]

        content_nodes = []
        content_nodes.append({"tag": "h3", "children": ["AI 통합 리포트"]})

        for raw_line in text_body.split("\n"):
            line = raw_line.strip()
            if not line:
                continue 

            if line.startswith("### "):
                content_nodes.append({
                    "tag": "h4",
                    "children": [line[4:]]
                })
            elif line.startswith("[") and line.endswith("]"):
                content_nodes.append({
                    "tag": "p",
                    "children": [{
                        "tag": "b",
                        "children": [line]
                    }]
                })
            else:
                content_nodes.append({
                    "tag": "p",
                    "children": [line]
                })

        data = {
            "access_token": token,
            "title": title,
            "content": json.dumps(content_nodes),
            "return_content": False,
        }

        telegraph_create_page_url = "https://api.telegra.ph/createPage"
        resp = requests.post(telegraph_create_page_url, data=data, timeout=10).json()

        if resp.get("ok"):
            return resp["result"]["url"]
        else:
            print(f"Telegraph API 오류: {resp.get('error')}")
            return ""
    except Exception as e:
        print(f"Telegraph 생성 실패: {e}")
        return ""


# ----------------------------------------
# [Part 5] 텔레그램 전송 (기존 동일)
# ----------------------------------------
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARNING] 텔레그램 토큰 설정 누락. 전송 생략.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    def split_message(msg: str, chunk_size: int = 4000) -> list[str]:
        chunks = []
        current = []
        current_len = 0
        for line in msg.splitlines(keepends=True):
            if len(line) >= chunk_size:
                if current:
                    chunks.append("".join(current))
                    current = []
                    current_len = 0
                for i in range(0, len(line), chunk_size):
                    chunks.append(line[i : i + chunk_size])
                continue
            if current_len + len(line) > chunk_size:
                chunks.append("".join(current))
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += len(line)
        if current:
            chunks.append("".join(current))
        return chunks

    chunks = split_message(message, chunk_size=4000)

    for i, chunk_text in enumerate(chunks):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, data=payload, timeout=10)
            if resp.status_code != 200:
                print(f"[ERROR] 텔레그램 전송 실패 ({i}): {resp.text}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] 텔레그램 요청 중 예외: {e}")

    print("[INFO] 텔레그램 메시지 전송 완료")


# ----------------------------------------
# 메인 실행
# ----------------------------------------
def main():
    # 1. 링크 수집
    links = collect_naver_news_links()
    if not links:
        print("수집된 기사가 없습니다.")
        return

    stats = {}
    for item in links:
        stats[item["source"]] = stats.get(item["source"], 0) + 1
    header_stats = " | ".join([f"{k} {v}" for k, v in stats.items()])
    safe_header_stats = escape_html(header_stats)

    # 2. 본문 크롤링
    contents = fetch_contents_parallel(links)

    # 3. Gemini 분석
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY가 없어 분석을 생략합니다.")
        return

    result = analyze_with_gemini(contents)

    # 4. 리포트 생성
    today_str = get_kst_today()
    telegram_msg = f"<b>🗞 {today_str} 신문 1면 브리핑 (Powered by Gemini)</b>\n\n"
    telegram_msg += f"📊 <b>수집 현황:</b> {safe_header_stats}\n\n"
    webview_text = f"📰 {today_str} 신문 1면 통합 리포트\n\n[수집 현황] {header_stats}\n\n"

    topics = result.get("topics", [])
    topics.sort(key=lambda t: len(t.get("ids", [])), reverse=True)

    if not topics:
        telegram_msg += "<b>⚠️ 리포트 생성 실패: 분석 결과가 없습니다.</b>"
        webview_text = "리포트 생성 실패"
    else:
        for topic in topics:
            title = topic.get("title", "무제")
            ids = topic.get("ids", [])
            bullets = topic.get("summary_bullets", [])
            full_article = topic.get("full_article", "")
            press_critiques = topic.get("press_critiques", [])

            # 텔레그램 메시지
            telegram_msg += f"━━━━━━━━━━━━━━\n"
            telegram_msg += f"📌 <b>{escape_html(title)}</b> ({len(ids)}건)\n"
            
            link_tags = []
            for idx in ids:
                if idx < len(contents):
                    item = contents[idx]
                    link_tags.append(
                        f"<a href=\"{escape_html(item['url'])}\">{escape_html(item['source'])}</a>"
                    )
            telegram_msg += f"🔗 {' , '.join(link_tags)}\n\n"

            for bullet in bullets:
                telegram_msg += f"• {escape_html(bullet)}\n"
            telegram_msg += "\n"

            if press_critiques:
                telegram_msg += "📰 <b>언론사별 논조</b>\n"
                for pc in press_critiques:
                    src = pc.get("source", "")
                    pos = pc.get("position", "")
                    if src and pos:
                        telegram_msg += f"- {escape_html(src)}: {escape_html(pos)}\n"
                telegram_msg += "\n"

            # 웹뷰 텍스트
            webview_text += f"\n### 📌 {title} ({len(ids)}건)\n"
            webview_text += "\n[핵심 요약]\n"
            for bullet in bullets:
                webview_text += f" - {bullet}\n"
            
            webview_text += "\n[통합 심층 기사]\n"
            webview_text += f"{full_article}\n"

            if press_critiques:
                webview_text += "\n[언론사별 비판/논조]\n"
                for pc in press_critiques:
                    src = pc.get("source", "")
                    pos = pc.get("position", "")
                    tone = pc.get("tone", "")
                    webview_text += f" - {src}: ({tone}) {pos}\n"
            webview_text += "\n\n"

    # 5. Telegraph 링크 생성 및 전송
    webview_url = create_telegraph_simple(f"{today_str} 조간 브리핑", webview_text)

    if webview_url:
        telegram_msg += f"\n\n📱 <b><a href='{webview_url}'>👉 전체 리포트 크게 보기</a></b>"
    
    send_telegram(telegram_msg)

if __name__ == "__main__":
    main()
