import os
import time
import json
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------------------
# 환경 변수 로드 (.env는 로컬용, GitHub Actions에서는 env 로딩)
# ----------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. OPENAI_API_KEY를 설정하세요.")

client = OpenAI(api_key=api_key)


def is_test_mode() -> bool:
    """
    TEST_MODE 환경변수(true/1/yes/on)이면 테스트 모드.
    - 테스트 모드: gpt-5-nano (최저 비용)
    - 일반 모드: 요약 gpt-5-mini, 분석 gpt-5.1 (고품질)
    """
    val = os.getenv("TEST_MODE", "false").strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def log(msg: str) -> None:
    """단순 로그 함수 (앞에 시간 붙여서 보기 좋게)"""
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ----------------------------------------
# 1) urls.txt에서 (언론사, URL) 목록 읽기
#    '# 동아일보' 같은 줄을 언론사 이름으로 사용
# ----------------------------------------
def load_items_from_file(path: str) -> List[Dict[str, Optional[str]]]:
    items: List[Dict[str, Optional[str]]] = []
    current_source: Optional[str] = None
    idx = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # '# 언론사명' → 현재 언론사 이름으로 사용
            if line.startswith("#"):
                name = line[1:].strip()
                current_source = name if name else None
                continue

            # URL 줄
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
    resp = requests.get(url, headers=headers, timeout=20)
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
        "div#articeBody",
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
# 3) 기사 1개 요약 (테스트 모드 지원)
# ----------------------------------------
def summarize_article(source: str, url: str, text: str, test_mode: bool) -> str:
    if test_mode:
        model = "gpt-5-nano"
    else:
        model = "gpt-5-mini"

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

    log(f"  ↳ OpenAI 요약 요청 시작 (모델={model})")
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    dt = time.perf_counter() - t0
    log(f"  ↳ 요약 완료 (소요 {dt:.1f}초)")

    try:
        return resp.output[0].content[0].text
    except Exception:
        return getattr(resp, "output_text", str(resp))


# ----------------------------------------
# 4) 여러 기사 요약을 비교 분석 (테스트 모드 지원)
#    2번 항목 포맷 변경: "주제별 요약"만 남기고 그 안에 언론사+톤/프레임 녹여쓰기
# ----------------------------------------
def compare_summaries(summary_items: List[Dict], test_mode: bool) -> str:
    """
    summary_items: [
      { "index": 1, "source": "동아일보", "url": "...", "summary": "..." },
      ...
    ]
    """
    if test_mode:
        model = "gpt-5-nano"
    else:
        model = "gpt-5.1"

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

다음 네 가지 작업을 수행하라.

1) 모든 기사에서 **공통으로 등장하는 핵심 사실**을 정리하라.
   - 사건·정책·인물·숫자를 중심으로 bullet 형태로 정리

2) 기사를 내용에 따라 3~6개의 **주제**로 나누고, 각 주제마다 '주제별 요약'만 작성하라.
   - 예를 들어 (실제 주제 이름은 네가 판단해서 정하라):
     - A) 관세·대기업 투자·국내 산업/일자리
     - B) 대장동·검찰·검사장 징계/인사
     - C) 한·미 동맹·방위비·조인트 팩트시트
     - D) 청년 고용·주거·세대 격차
     - E) 생성형 AI·기술·연구윤리
   - 각 주제 아래에는 **다음 형식으로만** 서술하라.
     - "A 주제: (주제 이름)"
       - 동아/조선/중앙은 ~~~한 톤·프레임으로 다룬다.
       - 경향/한겨레는 ~~~한 톤·프레임으로 다룬다.
       - 한국일보(또는 기타 중도 매체)는 ~~~ 식으로 다룬다.
   - 즉, 주제별로 **각 언론사의 톤과 프레임을 한 단락 안에 녹여서** 요약하되,
     - "1) 이 주제를 다룬 기사와 언론사", "2) 각 언론의 톤·프레임 비교" 같은 하위 번호 제목은 쓰지 말 것
     - 개별 기사 번호([기사 1] 등)를 다시 언급하지 말 것

3) 언론사 이름(동아일보, 조선일보, 한겨레, 경향신문, 중앙일보, 한국일보 등)을 참고하여,
   전체적으로 **보수/진보/중도/경제지** 등 언론 지형이 어떻게 갈려 있는지 해석하라.
   - 각 언론이 위에서 정리한 주제들에 대해 어떤 패턴으로 반응하는지
     (친정부/반정부, 친시장/반시장, 동맹 강화/비용 비판 등)를 정리하라.

4) 마지막으로,
   독자가 이 기사들을 읽을 때
   **비판적으로 봐야 할 포인트 3~5가지**를 정리하라.
   - 가능하면 위에서 나눈 주제들과 연결해서,
     - 예: "대기업 투자 숫자 보도의 한계", "방위비 숫자 프레이밍", "노동·청년이 지워지는 방식" 등으로 구체적으로 써라.

[기사 요약들]
{joined}
""".strip()

    log(f"[단계] 최종 비교·분석용 OpenAI 호출 시작 (모델={model})")
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "너는 한국어로 답변하는 미디어 비평가다."},
            {"role": "user", "content": prompt},
        ],
    )
    dt = time.perf_counter() - t0
    log(f"[단계] 최종 비교·분석 생성 완료 (소요 {dt:.1f}초)")

    try:
        return resp.output[0].content[0].text
    except Exception:
        return getattr(resp, "output_text", str(resp))


# ----------------------------------------
# 4-1) 링크 섹션을 위한 "주제별 분류" (초저비용 gpt-5-nano)
#      → { "A. 관세·대기업 투자·국내 산업/일자리": [1,2,6,...], ... } 형태
# ----------------------------------------
def classify_topics(summary_items: List[Dict]) -> Dict[str, List[int]]:
    """
    요약들을 바탕으로 기사들을 3~6개 주제로 묶어 주제별 링크 섹션을 만들기 위한 분류.
    응답 형식 (JSON만 반환하도록 강하게 요구):
    {
      "A. 관세·대기업 투자·국내 산업/일자리": [1, 2, 5],
      "B. 대장동·검찰·검사장 징계/인사": [3, 7],
      ...
    }
    실패 시 빈 dict 반환.
    """
    model = "gpt-5-nano"

    blocks = []
    for item in summary_items:
        idx = item["index"]
        source = item["source"]
        summary = item["summary"]
        blocks.append(
            f"[기사 {idx}] 언론사: {source}\n요약:\n{summary}\n"
        )
    joined = "\n\n".join(blocks)

    prompt = f"""
너는 기사 요약들을 주제별로 묶는 분류기다.

아래에 [기사 N] 형식으로 여러 기사 요약이 주어진다.
각 기사를 내용 기준으로 3~6개의 주제로 묶어라.

반환 형식은 반드시 **JSON만** 사용해야 한다.
JSON 오브젝트의 key는 "A. 주제 이름" 같은 문자열,
value는 그 주제에 속하는 기사 번호(N)의 정수 배열이다.

예시:
{{
  "A. 관세·대기업 투자·국내 산업/일자리": [1, 2, 5],
  "B. 대장동·검찰·검사장 징계/인사": [3, 7],
  "C. 한·미 동맹·방위비·조인트 팩트시트": [4, 6],
  "D. 청년 고용·주거·세대 격차": [8],
  "E. 생성형 AI·기술·연구윤리": [9]
}}

주의:
- 반드시 유효한 JSON만 출력하라.
- JSON 바깥에 다른 텍스트(설명, 코드블록 등)를 절대 넣지 마라.
- 기사 번호는 [기사 N]의 N 값을 사용하라.

[기사 요약들]
{joined}
""".strip()

    log("[단계] 주제별 링크 구성을 위한 분류 호출 시작 (모델=gpt-5-nano)")
    try:
        t0 = time.perf_counter()
        resp = client.responses.create(
            model=model,
            input=prompt,
        )
        dt = time.perf_counter() - t0
        log(f"[단계] 주제 분류 응답 수신 (소요 {dt:.1f}초)")

        try:
            text = resp.output[0].content[0].text
        except Exception:
            text = getattr(resp, "output_text", "")

        data = json.loads(text)
        # JSON 구조가 맞는지 간단 체크
        if not isinstance(data, dict):
            raise ValueError("JSON 최상위가 dict가 아님")
        return {
            str(k): [int(n) for n in v]
            for k, v in data.items()
            if isinstance(v, list)
        }
    except Exception as e:
        log(f"[경고] 주제 분류 실패, 기본 링크 리스트 형식으로 대체: {e}")
        return {}


# ----------------------------------------
# 5) 텔레그램으로 결과 전송
#    - 길이 4096자 제한을 고려해 여러 메시지로 나눠서 전송
# ----------------------------------------
def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        log("[경고] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않아 텔레그램 전송을 생략합니다.")
        return

    base_url = f"https://api.telegram.org/bot{token}/sendMessage"

    # 텔레그램 메시지는 최대 4096자 → 조금 여유 있게 3500자로 나눔
    chunk_size = 3500

    # 줄 단위로 나눠서 청크 끊기 (HTML/마크업 안 쓰고 순수 텍스트이므로 안전)
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    current = ""
    for line in lines:
        if len(current) + len(line) > chunk_size and current:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)

    total = len(chunks)
    log(f"[단계] 텔레그램 전송 시작 (총 {total}개 청크)")
    for i, chunk in enumerate(chunks, start=1):
        if total > 1:
            header = f"[신문 1면 분석 리포트 {i}/{total}]\n\n"
        else:
            header = "[신문 1면 분석 리포트]\n\n"

        payload = {
            "chat_id": chat_id,
            "text": header + chunk,
        }

        try:
            resp = requests.post(base_url, data=payload, timeout=20)
            resp.raise_for_status()
            log(f"  → 텔레그램 전송 완료 ({i}/{total})")
        except Exception as e:
            log(f"  [에러] 텔레그램 전송 실패 ({i}/{total}): {e}")
            break

        time.sleep(0.5)  # 너무 빠른 연속 전송 방지


# ----------------------------------------
# 6) 전체 파이프라인
# ----------------------------------------
def run_pipeline(
    url_file: str = "urls.txt",
    out_dir: str = "output_frontpage",
) -> None:
    start_all = time.perf_counter()
    test_mode = is_test_mode()

    log("========================================")
    log("신문 1면 분석 파이프라인 시작")
    log(f"TEST_MODE = {test_mode}")
    log("========================================")

    if not os.path.exists(url_file):
        raise SystemExit(f"URL 파일을 찾을 수 없습니다: {url_file}")

    items = load_items_from_file(url_file)
    if not items:
        raise SystemExit("URL이 한 개도 없습니다. urls.txt를 확인하세요.")

    total = len(items)
    log(f"[단계] URL 로드 완료: 총 {total}개 기사")

    os.makedirs(out_dir, exist_ok=True)

    summary_items: List[Dict] = []

    # 1단계: 각 기사 크롤링 + 요약
    for item in items:
        idx = item["index"]
        source = item["source"]
        url = item["url"]

        log(f"[기사 {idx}/{total}] [{source}] 본문 크롤링 시작")
        try:
            t0 = time.perf_counter()
            article_text = fetch_article_text(url)
            dt = time.perf_counter() - t0
            log(f"[기사 {idx}/{total}] 본문 크롤링 완료 (소요 {dt:.1f}초, 길이 {len(article_text)}자)")
        except Exception as e:
            log(f"[기사 {idx}/{total}] [에러] 기사 본문 추출 실패: {e}")
            continue

        log(f"[기사 {idx}/{total}] 요약 생성 시작")
        try:
            summary = summarize_article(source, url, article_text, test_mode=test_mode)
        except Exception as e:
            log(f"[기사 {idx}/{total}] [에러] 요약 생성 실패: {e}")
            continue

        summary_items.append(
            {
                "index": idx,
                "source": source,
                "url": url,
                "summary": summary,
            }
        )
        log(f"[기사 {idx}/{total}] 요약 완료")

        # 너무 빠른 연속 호출을 피하기 위한 약간의 딜레이
        time.sleep(0.3)

    if not summary_items:
        raise SystemExit("요약이 한 개도 생성되지 않았습니다.")

    log(f"[단계] 요약 생성 완료: {len(summary_items)}/{total}개 기사")

    # 2단계: 요약들을 기반으로 최종 비교·분석
    log("[단계] 최종 비교·분석 생성 단계로 진입")
    analysis_body = compare_summaries(summary_items, test_mode=test_mode)

    # 3단계: 링크 섹션 생성을 위한 주제 분류
    topic_map = classify_topics(summary_items)
    idx_map = {item["index"]: item for item in summary_items}

    # 모드 정보 헤더
    mode_label = "테스트 (저비용: gpt-5-nano)" if test_mode else "일반 (고품질: 요약 gpt-5-mini, 분석 gpt-5.1)"
    header = f"[모드] {mode_label}\n"

    # 링크 목록 생성 (주제별)
    link_lines: List[str] = []
    if topic_map:
        link_lines.append("[기사 링크 모음 (주제별)]")
        for topic_label in sorted(topic_map.keys()):
            link_lines.append(f"{topic_label}")
            # 같은 주제 안에서 언론사별로 묶기
            by_source: Dict[str, List[str]] = {}
            for idx in topic_map[topic_label]:
                item = idx_map.get(idx)
                if not item:
                    continue
                source = item["source"]
                url = item["url"]
                by_source.setdefault(source, []).append(url)
            for source, urls in by_source.items():
                # 한 줄에 같은 언론사의 URL들을 , 로 붙여서 표현
                joined_urls = ", ".join(urls)
                link_lines.append(f"- {source}: {joined_urls}")
            link_lines.append("")  # 주제 간 빈 줄
    else:
        # 분류 실패 시: 기존처럼 평탄한 리스트
        link_lines.append("[기사 링크 모음]")
        for item in summary_items:
            idx = item["index"]
            source = item["source"]
            url = item["url"]
            link_lines.append(f"{idx}. [{source}] {url}")

    links_section = "\n".join(link_lines)
    links_section = "\n\n----------\n" + links_section

    final_report = header + "\n" + analysis_body + links_section

    # 최종 리포트 파일 저장
    report_path = os.path.join(out_dir, "final_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    log(f"[단계] 최종 리포트 파일 저장: {report_path}")

    # 텔레그램으로도 전송
    send_telegram_message(final_report)

    total_dt = time.perf_counter() - start_all
    log(f"[완료] 전체 파이프라인 종료 (총 소요 {total_dt:.1f}초)")


if __name__ == "__main__":
    run_pipeline()
