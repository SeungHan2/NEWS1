import os
import time
import json
import requests
# timezone 임포트 추가 (Python 버전 호환성 확보)
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from dotenv import load_dotenv

# ----------------------------------------
# 환경 변수 및 설정
# ----------------------------------------
load_dotenv()

# .strip()을 추가하여 토큰이나 ID 앞뒤의 모든 공백/특수 문자를 제거합니다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 모델명은 사용자의 로그에서 확인된 'gemini-2.0-flash'로 고정
GEMINI_MODEL_NAME = 'gemini-2.0-flash' 

PRESS_LIST: List[Tuple[str, str]] = [
    ("동아일보", "020"),
    ("한국일보", "469"),
    ("조선일보", "023"),
    ("중앙일보", "025"),
    ("한겨레", "028"),
    ("경향신문", "032"),
]
# URL 조합을 f-string으로 명시적으로 처리하기 위해 사용하지 않음
# BASE_NEWPAPER_URL = "https://media.naver.com/press/{press}/newspaper?date={date}"

# ----------------------------------------
# [Part 1] 네이버 1면 링크 수집
# ----------------------------------------
def get_kst_today() -> str:
    # timezone.utc를 사용하여 Python 버전에 관계없이 UTC를 명확하게 지정
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc + timedelta(hours=9)
    return now_kst.strftime("%Y%m%d")

def fetch_html(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    # URL 오류 방지를 위해 strip() 적용
    resp = requests.get(url.strip(), headers=headers, timeout=20) 
    resp.raise_for_status()
    return resp.text

def extract_a1_links(html: str, page_url: str, press_code: str, date: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if f"/article/newspaper/{press_code}/" not in href: continue
        if f"date={date}" not in href: continue
        full_url = urljoin(page_url, href)
        
        is_a1 = False
        parent = a
        for _ in range(6):
            parent = parent.parent
            if parent is None: break
            text = parent.get_text(" ", strip=True)
            if any(key in text for key in ["A1면", "A01면", "1면", "1 面"]):
                is_a1 = True
                break
        if is_a1: candidates.append(full_url)

    if not candidates: # Fallback
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/article/newspaper/{press_code}/" in href and f"date={date}" in href:
                full_url = urljoin(page_url, href)
                if full_url not in seen:
                    candidates.append(full_url)
                    seen.add(full_url)
            if len(candidates) >= 4: break
    return list(set(candidates))

def collect_naver_news_links() -> List[Dict[str, str]]:
    date = get_kst_today()
    print(f"[INFO] {date}일자 1면 기사 수집 시작")
    all_items = []
    for press_name, press_code in PRESS_LIST:
        url = "" # url 변수 초기화
        try:
            # f-string을 사용해 명확하게 URL 조합 (이전 오류 해결 코드 반영)
            url = f"https://media.naver.com/press/{press_code}/newspaper?date={date}".strip()
            
            html = fetch_html(url)
            links = extract_a1_links(html, url, press_code, date)
            for link in links:
                all_items.append({"source": press_name, "url": link})
        except Exception as e:
            # 에러 로그 출력 시 URL을 같이 출력
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
            "content": content[:4000] if content else "본문 없음" # 길이 제한
        }
    except:
        return item

def fetch_contents_parallel(items: list) -> list:
    print(f"[INFO] 총 {len(items)}개 기사 본문 크롤링 중...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_single_article_content, items))
    return results

# ----------------------------------------
# [Part 3] Gemini 분석 (리포트 작성)
# ----------------------------------------
def analyze_with_gemini(articles: list) -> dict:
    print(f"[INFO] {GEMINI_MODEL_NAME} 분석 요청 시작...")
    
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config={"response_mime_type": "application/json"}
    )

    articles_text = ""
    for i, art in enumerate(articles):
        articles_text += f"[ID:{i}] 언론사:{art['source']} | 내용:{art['content'][:2000]}\n"

    # 통합 기사 분량 및 상세 요구사항 강화 프롬프트
    prompt = f"""
    너는 전문 뉴스 에디터다. 오늘자 신문 1면 기사들을 종합하여 고품질 리포트를 작성하라.
    
    [요구사항]
    1. 기사들을 유사한 주제(정치, 경제, 사회 등)로 그룹화하라.
    2. **주제별 통합 기사 작성**: 각 주제에 대해 개별 기사를 단순히 나열하지 말고, 모든 내용을 종합하여 **하나의 완결된 심층 기사**로 새로 써라.
        - **분량**: 반드시 **최소 500자 이상**의 상세한 글로 작성할 것.
        - **구성**: 기사의 배경, 현재 상황, 언론사별 주요 주장, 그리고 향후 전망이나 전문가 분석 등 다각도의 관점을 포함하여 작성할 것.
        - **톤**: 전문가가 작성한 객관적인 논조의 기사 형태를 유지할 것.
    3. **요약본(Bullets)**: 바쁜 독자를 위해, 통합 기사의 내용을 3줄 이내의 핵심 단문(Bullet point)으로 요약하라.
    4. 결과는 반드시 JSON 형식이어야 한다.

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
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # JSON 응답을 감싸는 마크다운 코드 블록 제거
        if raw_text.startswith('```json'):
            raw_text = raw_text.removeprefix('```json').removesuffix('```').strip()
        
        return json.loads(raw_text)
        
    except json.JSONDecodeError as e:
        # JSON 디코딩 실패 시: 모델이 생성한 원본 텍스트를 출력 (디버깅용)
        print(f"[CRITICAL ERROR] JSON 디코딩 실패: {e}")
        print("--- Gemini Raw Output Start ---")
        if response:
            print(response.text)
        else:
            print("No response object available.")
        print("--- Gemini Raw Output End ---")
        return {"topics": []}
    
    except Exception as e:
        print(f"[CRITICAL ERROR] Gemini 분석 중 기타 에러 발생: {e}")
        return {"topics": []}

# ----------------------------------------
# [Part 4] Telegraph 페이지 생성 (웹뷰)
# ----------------------------------------
def create_telegraph_simple(title: str, text_body: str) -> str:
    """간단한 텍스트 기반 Telegraph 페이지 생성"""
    try:
        # 1. 토큰 생성: URL 깨끗하게 유지 (수정됨)
        telegraph_account_url = "[https://api.telegra.ph/createAccount?short_name=NewsAI](https://api.telegra.ph/createAccount?short_name=NewsAI)"
        print(f"[DEBUG] Telegraph Account URL: {telegraph_account_url}")
        
        r = requests.get(telegraph_account_url).json()
        token = r['result']['access_token']
        
        content_nodes = []
        content_nodes.append({"tag": "h3", "children": ["AI 통합 리포트"]})
        
        current_p_children = []
        for line in text_body.split('\n'):
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
            "return_content": False
        }
        # 2. 페이지 생성: URL 깨끗하게 유지 (수정됨)
        telegraph_create_page_url = "[https://api.telegra.ph/createPage](https://api.telegra.ph/createPage)"
        resp = requests.post(telegraph_create_page_url, data=data).json()
        
        if resp.get('ok'):
            return resp['result']['url']
        else:
            print(f"Telegraph API 오류: {resp.get('error')}")
            return ""
    except Exception as e:
        # 이 시점에서 InvalidSchema가 발생하면 Telegraph URL 자체의 문자열 문제일 가능성이 100%
        print(f"Telegraph 생성 실패: {e}")
        return ""

# ----------------------------------------
# [Part 5] 텔레그램 전송 (HTML 모드)
# ----------------------------------------
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: 
        print("[WARNING] 텔레그램 토큰 또는 채팅 ID가 없어 전송을 건너뜁니다.")
        return
        
    # URL 구성: URL 깨끗하게 유지 (수정됨)
    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # 🚨 디버깅 코드 추가: URL 길이를 출력하고, 토큰이 삽입된 URL의 앞부분을 확인
    # 토큰에 문제가 있다면 URL 길이가 비정상적이거나, URL에 이상한 문자가 보일 수 있음.
    # 안전을 위해 토큰 부분은 *로 마스킹하여 출력
    masked_url = url.replace(TELEGRAM_BOT_TOKEN, "***masked***")
    print(f"[DEBUG] Telegram URL length: {len(url)}")
    print(f"[DEBUG] Telegram URL fragment (masked): {masked_url[:70]}")
    
    chunk_size = 4000 
    for i in range(0, len(message), chunk_size):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": message[i:i+chunk_size], 
            "parse_mode": "HTML", 
            "disable_web_page_preview": True 
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

    # 언론사별 수량 카운트
    stats = {}
    for item in links:
        stats[item['source']] = stats.get(item['source'], 0) + 1
    
    # 통계 헤더 생성
    header_stats = " | ".join([f"{k} {v}" for k, v in stats.items()])

    # 2. 본문 크롤링
    contents = fetch_contents_parallel(links)

    # 3. Gemini 분석
    if not GEMINI_API_KEY: 
        print("API 키가 없어 분석을 생략합니다.")
        return
    
    result = analyze_with_gemini(contents)
    
    # 4. 리포트 및 웹뷰 컨텐츠 생성
    today_str = get_kst_today()
    
    # 텔레그램용 메시지 (요약 위주)
    telegram_msg = f"<b>🗞 {today_str} 신문 1면 브리핑</b>\n\n"
    telegram_msg += f"📊 <b>수집 현황:</b> {header_stats}\n\n"
    
    # 웹뷰용 전체 텍스트
    webview_text = f"📰 {today_str} 신문 1면 통합 리포트\n\n[수집 현황] {header_stats}\n\n"

    topics = result.get("topics", [])
    
    # === [요청 사항 반영: 주제별 기사 수에 따라 내림차순 정렬] ===
    # 'ids' 리스트의 길이를 기준으로 내림차순 정렬
    topics.sort(key=lambda t: len(t.get('ids', [])), reverse=True)
    # =========================================================
    
    if not topics:
        telegram_msg += "<b>⚠️ 리포트 생성 실패: 분석 과정에서 오류가 발생했거나, AI가 답변을 거부했습니다. GitHub Actions 로그를 확인하세요.</b>"
        webview_text = "리포트 생성 실패"
    else:
        for topic in topics:
            title = topic.get('title', '무제')
            ids = topic.get('ids', [])
            bullets = topic.get('summary_bullets', [])
            full_article = topic.get('full_article', '')

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
    
    # 텔레그램 메시지 하단에 링크 추가
    if webview_url:
        telegram_msg += f"\n\n📱 <b><a href='{webview_url}'>👉 전체 리포트 크게 보기 (Safari/Web)</a></b>"

    # 6. 전송
    print("[INFO] 텔레그램 전송 중...")
    send_telegram(telegram_msg)
    print("[INFO] 완료.")

if __name__ == "__main__":
    main()
