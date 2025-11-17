import os
import time
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------------------
# 환경 변수 로드 (.env 지원)
# ----------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일 또는 환경변수를 확인하세요.")

client = OpenAI(api_key=api_key)


# ----------------------------------------
# 1) urls.txt에서 (언론사, URL) 목록 읽기
#    - '# 동아일보' 같은 주석 줄을 언론사 이름으로 사용
# ----------------------------------------
def load_items_from_file(path: str) -> List[Dict[str, Optional[str]]]:
    """
    urls.txt 예시:

    # 동아일보
    https://n.news.naver.com/article/newspaper/020/0003674837?date=20251117
    ...

    # 한국일보
    https://...

    반환 형식:
    [
      {"index": 1, "source": "동아일보", "url": "https://..."},
      {"index": 2, "source": "동아일보", "url": "https://..."},
      {"index": 5, "source": "한국일보", "url": "https://..."},
      ...
    ]
    """
    items: List[Dict[str, Optional[str]]] = []
    current_source: Optional[str] = None
    idx = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # '# 언론사명' → 현재 언론사 설정
            if line.startswith("#"):
                name = line[1:].strip()
                current_source = name if name else None
                continue

            # 이 줄은 URL이라고 가정
            idx += 1
            items.append(
                {
                    "index": idx,
                    "source": current_source or "언론사 미상",
                    "url": line,
                }
            )

    return items


# ----------------------------------------
# 2) 뉴스 URL에서 기사 본문 텍스트 추출
# ----------------------------------------
def fetch_article_text(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    candidates = []

    # 공통 article 태그
    article = soup.find("article")
    if article:
        candidates.append(article)

    # 네이버 / 주요 신문에서 자주 쓰는 본문 컨테이너들
    selectors = [
        "div#dic_area",            # 네이버 뉴스
        "div#newsEndContents",     # 네이버 구형
        "div.newsct_article",      # 네이버 신형
        "div#articeBody",          # 일부 신문
        "div.article_body",
        "div.article-body",
        "div#articleBodyContents",
        "div#article-view-content-div",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            candidates.append(node)

    # schema.org articleBody
    node = soup.select_one("[itemprop='articleBody']")
    if node:
        candidates.append(node)

    root = candidates[0] if candidates else (soup.body or soup)

    paragraphs: List[str] = []
    for p in root.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)

    if not paragraphs:
        return root.get_text("\n", strip=True)

    return "\n".join(paragraphs)


# ----------------------------------------
# 3) 기사 1개 요약 (gpt-5-mini 사용)
# ----------------------------------------
def summarize_article(source: str, url: str, text: str) -> str:
    prompt = f"""
아래는 한국 신문 1면 기사 전문이다.

[언론사] {source}
[URL] {url}

이 기사를 다음 기준에 따라 한국어로 10줄 이내로 요약하라.

1) 사건·정책·인물 등 핵심 사실을 정리 (객관적 팩트 위주)
2) 기사에서 사용하는 표현과 구성에 기반하여 **논조(톤)**를 설명
3) 중요한 숫자·날짜·고유명사는 가능하면 그대로 남긴다.

[기사 본문]
{text}
""".strip()

    resp = client.responses.create(
        model="gpt-5-mini",  # ← 요약 단계는 가성비 좋은 mini 모델
        input=prompt,
    )

    try:
        return resp.output[0].content[0].text
    except Exception:
        return getattr(resp, "output_text", str(resp))


# ----------------------------------------
# 4) 여러 기사 요약을 비교 분석 (gpt-5.1 사용)
# ----------------------------------------
def compare_summaries(summary_items: List[Dict]) -> str:
    """
    summary_items: [
      { "index": 1, "source": "동아일보", "url": "...", "summary": "..." },
      ...
    ]
    """
    blocks = []
    for item in summary_items:
        idx = item["index"]
        source = item["source"]
        url = item["url"]
        summary = item["summary"]
        blocks.append(
            f"[기사 {idx}] 언론사: {source}\nURL: {url}\n요약:\n{summary}\n"
        )

    joined = "\n\n".join(blocks)

    prompt = f"""
아래는 서로 다른 언론사 1면 기사들의 요약이다.
각 [기사 N]은 하나의 기사에 대응하며, '언론사' 정보가 포함되어 있다.

다음 작업을 수행하라.

1) 모든 기사에서 **공통으로 등장하는 핵심 사실**을 정리하라.
   - 사건·정책·인물·숫자를 중심으로 bullet 형태로 정리

2) 기사별로 **논조(톤)와 프레임**을 비교하라.
   - [기사 N] 단위로 정리
   - 각 언론사가 정부/야당/대기업/서민/국제사회 등을 어떻게 묘사하는지 설명
   - 어떤 단어 선택과 구성으로 프레임을 만드는지 구체적으로 적는다.

3) 언론사 이름(동아일보, 조선일보, 한겨레, 경향신문, 중앙일보, 한국일보 등)을 참고하여,
   전체적으로 **보수/진보/중도/경제지** 등 언론 지형이 어떻게 갈려 있는지 해석하라.

4) 마지막으로,
   독자가 이 기사들을 읽을 때
   **비판적으로 봐야 할 포인트 3~5가지**를 정리하라.

[기사 요약들]
{joined}
""".strip()

    resp = client.responses.create(
        model="gpt-5.1",  # ← 최종 분석은 가장 높은 품질의 모델 사용
        input=[
            {"role": "system", "content": "너는 한국어로 답변하는 미디어 비평가다."},
            {"role": "user", "content": prompt},
        ],
    )

    try:
        return resp.output[0].content[0].text
    except Exception:
        return getattr(resp, "output_text", str(resp))


# ----------------------------------------
# 5) 전체 파이프라인
#    - 기사/요약은 파일로 저장하지 않고
#      최종 리포트만 final_report.txt로 저장
# ----------------------------------------
def run_pipeline(
    url_file: str = "urls.txt",
    out_dir: str = "output_frontpage",
) -> None:
    if not os.path.exists(url_file):
        raise SystemExit(f"URL 파일을 찾을 수 없습니다: {url_file}")

    items = load_items_from_file(url_file)
    if not items:
        raise SystemExit("URL이 한 개도 없습니다. urls.txt를 확인하세요.")

    os.makedirs(out_dir, exist_ok=True)

    summary_items: List[Dict] = []

    # 1단계: 각 기사 크롤링 + 요약
    total = len(items)
    for item in items:
        idx = item["index"]
        source = item["source"]
        url = item["url"]

        print(f"[+] ({idx}/{total}) 기사 크롤링 및 요약 중: [{source}] {url}")

        try:
            article_text = fetch_article_text(url)
        except Exception as e:
            print(f"    [에러] 기사 본문 추출 실패: {e}")
            continue

        try:
            summary = summarize_article(source, url, article_text)
        except Exception as e:
            print(f"    [에러] 요약 생성 실패: {e}")
            continue

        summary_items.append(
            {
                "index": idx,
                "source": source,
                "url": url,
                "summary": summary,
            }
        )

        # 너무 빠른 연속 호출을 피하기 위한 약간의 딜레이
        time.sleep(0.3)

    if not summary_items:
        raise SystemExit("요약이 한 개도 생성되지 않았습니다.")

    # 2단계: 요약들을 기반으로 최종 비교·분석
    print("\n[+] 요약들을 기반으로 최종 비교·분석 생성 중...")

    final_report = compare_summaries(summary_items)

    report_path = os.path.join(out_dir, "final_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"[완료] 최종 리포트 저장: {report_path}")


if __name__ == "__main__":
    run_pipeline()
