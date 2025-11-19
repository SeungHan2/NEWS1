import os
import time
import json
import requests
from datetime import datetime, timedelta
from urllib.parse import urljoin
from typing import List, Tuple, Dict, Optional
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from dotenv import load_dotenv

# ----------------------------------------
# 환경 변수 및 설정
# ----------------------------------------
load_dotenv() # 로컬 테스트용 (.env 파일 로드)

# GitHub Actions에서는 Secrets에서 주입됨
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not GEMINI_API_KEY:
    print("[경고] GEMINI_API_KEY가 없습니다. (로컬 테스트가 아니라면 GitHub Secrets 확인 필요)")

# Gemini 설정
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Naver 신문사별 코드 (제공해주신 코드 활용)
PRESS_LIST: List[Tuple[str, str]] = [
    ("동아일보", "020"),
    ("한국일보", "469"),
    ("조선일보", "023"),
    ("중앙일보", "025"),
    ("한겨레", "028"),
    ("경향신문", "032"),
]
BASE_NEWPAPER_URL = "https://media.naver.com/press/{press}/newspaper?date={date}"


# ----------------------------------------
# [Part 1] 네이버 1면 링크 수집 (Crawler)
# ----------------------------------------
def get_kst_today() -> str:
    """현재 KST(UTC+9) 기준 날짜를 YYYYMMDD로 반환"""
    now_utc = datetime.utcnow()
    now_kst = now_utc + timedelta(hours=9)
    return now_kst.strftime("%Y%m%d")

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

def extract_a1_links(html: str, page_url: str, press_code: str, date: str) -> List[str]:
    """A1(1면) 기사 링크 추출"""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if f"/article/newspaper/{press_code}/" not in href: continue
        if f"date={date}" not in href: continue

        full_url = urljoin(page_url, href)

        # 부모 쪽에 A1/1면 표시 있는지 확인
        is_a1 = False
        parent = a
        for _ in range(6):
            parent = parent.parent
            if parent is None: break
            text = parent.get_text(" ", strip=True)
            if any(key in text for key in ["A1면", "A01면", "A 1면", "A 01면", "1면", "1 面"]):
                is_a1 = True
                break

        if is_a1:
            candidates.append(full_url)

    # Fallback: A1 키워드 없으면 상위 4개 가져오기
    if not candidates:
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
    
    # 중복 제거
    return list(set(candidates))

def collect_naver_news_links() -> List[Dict[str, str]]:
    """모든 언론사의 1면 기사 링크를 수집하여 리스트로 반환"""
    date = get_kst_today()
    print(f"[INFO] {date}일자 1면 기사 수집 시작")
    
    all_items = []
    
    for press_name, press_code in PRESS_LIST:
        page_url = BASE_NEWPAPER_URL.format(press=press_code, date=date)
        try:
            html = fetch_html(page_url)
            links = extract_a1_links(html, page_url, press_code, date)
            print(f"  - {press_name}: {len(links)}개 발견")
            for link in links:
                all_items.append({"source": press_name, "url": link})
        except Exception as e:
            print(f"  [에러] {press_name} 수집 실패: {e}")
            
    return all_items

# ----------------------------------------
# [Part 2] 본문 크롤링 (Parallel Fetcher)
# ----------------------------------------
def fetch_single_article_content(item: dict) -> dict:
    """단일 기사 본문 추출"""
    url = item["url"]
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # 네이버 뉴스 본문 셀렉터 모음
        selectors = [
            "div#dic_area", "div#newsEndContents", "div.newsct_article",
            "div#articeBody", "div#articleBodyContents"
        ]
        content = ""
        for selector in selectors:
            node = soup.select_one(selector)
            if node:
                content = node.get_text("\n", strip=True)
                break
        
        return {
            "source": item["source"],
            "url": url,
            "content": content if content else "본문 추출 실패"
        }
    except Exception as e:
        return {"source": item["source"], "url": url, "content": f"에러: {e}"}

def fetch_contents_parallel(items: list) -> list:
    """ThreadPool로 빠르게 본문 긁어오기"""
    print(f"[INFO] 총 {len(items)}개 기사 본문 크롤링 중...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_single_article_content, items))
    return results

# ----------------------------------------
# [Part 3] Gemini 분석 및 리포트 생성
# ----------------------------------------
def analyze_with_gemini(articles: list) -> dict:
    print("[INFO] Gemini 1.5 Flash 분석 요청 시작...")
    
    # 모델명 수정: 'gemini-1.5-flash-latest' -> 'gemini-1.5-flash'
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash', 
        generation_config={"response_mime_type": "application/json"}
    )

    articles_text = ""
    for i, art in enumerate(articles):
        articles_text += f"[ID:{i}] {art['source']} - {art['content'][:3000]}\n" # 너무 길면 자름

    prompt = f"""
    오늘자 한국 주요 신문 1면 기사들이다. 
    이 내용들을 종합해 '오늘의 조간 브리핑'을 작성해라.

    [요구사항]
    1. 전체를 관통하는 핵심 이슈와 분위기 요약 (Markdown 형식)
    2. 주요 주제별(정치, 경제, 사회 등)로 기사들을 분류하고 각 주제에 대해 각 언론사의 논조(Tone)를 비교 분석하라.
    3. 반드시 JSON 형식으로만 답하라.

    [JSON 출력 형식]
    {{
        "report_body": "여기에 전체 리포트 본문(마크다운) 작성. 이모지 사용해서 가독성 높일 것.",
        "topics": [
            {{ "title": "주제A", "ids": [0, 1, 5] }},
            {{ "title": "주제B", "ids": [2, 3] }}
        ]
    }}

    [기사 목록]
    {articles_text}
    """

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"[에러] Gemini 분석 실패: {e}")
        return {"report_body": "분석에 실패했습니다.", "topics": []}

# ----------------------------------------
# [Part 4] 텔레그램 전송
# ----------------------------------------
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunk_size = 3500 # 텔레그램 제한 대비 여유있게

    for i in range(0, len(message), chunk_size):
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message[i:i+chunk_size], "parse_mode": "Markdown"}
        requests.post(url, data=data)
        time.sleep(0.5)

# ----------------------------------------
# 메인 실행 로직
# ----------------------------------------
def main():
    # 1. 링크 수집
    links = collect_naver_news_links()
    if not links:
        print("수집된 기사가 없습니다. 종료합니다.")
        return

    # 2. 본문 크롤링
    contents = fetch_contents_parallel(links)

    # 3. Gemini 분석
    if not GEMINI_API_KEY:
        print("API 키가 없어 분석을 생략합니다.")
        return
    
    result = analyze_with_gemini(contents)
    
    # 4. 리포트 조립
    final_report = f"🗞 *오늘의 신문 1면 브리핑* ({get_kst_today()})\n\n"
    final_report += result.get("report_body", "")
    
    final_report += "\n\n🔗 *관련 기사 원문*\n"
    for topic in result.get("topics", []):
        final_report += f"\n📌 *{topic['title']}*\n"
        
        # 해당 주제의 기사들 모으기
        topic_urls = {}
        for idx in topic['ids']:
            if idx < len(contents):
                item = contents[idx]
                src = item['source']
                if src not in topic_urls: topic_urls[src] = []
                topic_urls[src].append(item['url'])
        
        for src, urls in topic_urls.items():
            # 링크가 여러 개면 첫 번째만 대표로 표시하거나 나열
            final_report += f"- {src}: [기사보기]({urls[0]})\n"

    # 5. 전송
    print("[INFO] 텔레그램 전송 중...")
    send_telegram(final_report)
    print("[INFO] 완료.")

if __name__ == "__main__":
    main()
