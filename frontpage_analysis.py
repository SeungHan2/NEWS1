import os
import time
import json
import html
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
#    새 포맷: (2) 주제별 핵심 요약 → (3) 주제별 언론사별 중요 포인트
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

리포트는 다음 순서를 염두에 두고 작성하되,
(1. 주제별 기사 링크)는 코드에서 따로 생성하므로,
여기서는 **(2)와 (3)**만 생성하라.

[출력 구조]

[2] 주제별 핵심 요약
- 3~6개의 주제를 잡아서, 각 주제마다 다음 형식으로 정리한다.
  - "A 주제: (주제 이름)"
    - 이 주제를 관통하는 사건·정책·갈등의 핵심을 3~6줄 내로 간결하게 요약
    - 가능한 한 숫자·날짜·고유명사는 그대로 사용
    - 개별 언론사 이름은 여기에서는 언급하지 말고, 전체 흐름 기준으로만 정리

[3] 주제별 언론사별로 중요시하는 점
- 위에서 사용한 주제 이름과 동일한 순서/라벨(A,B,...)을 사용한다.
- 각 주제 아래, 다음 형식으로 **언론사별 톤·프레임·강조점**을 정리한다.
  - "A 주제: (주제 이름)"
    - 동아일보는 ~~~을 가장 강조하며, ~~~한 톤으로 다룬다.
    - 조선일보는 ~~~을 가장 강조하며, ~~~한 톤으로 다룬다.
    - 중앙일보는 ~~~을 가장 강조하며, ~~~한 톤으로 다룬다.
    - 한겨레는 ~~~을 가장 강조하며, ~~~한 톤으로 다룬다.
    - 경향신문은 ~~~을 가장 강조하며, ~~~한 톤으로 다룬다.
    - 한국일보는 ~~~을 가장 강조하며, ~~~한 톤으로 다룬다.
    - 그 밖의 언론사는, 공통적으로 ~~~을 강조하거나 ~~~을 생략한다 등으로 정리한다.
- 이때, 보수/진보/중도 같은 레이블로 언론사를 묶지 말고,
  **각 언론사가 기사 구성과 표현을 통해 무엇을 중요하게 보이게 했는지**를 추론해서 서술하라.
- 개별 기사 번호([기사 1] 등)는 다시 언급하지 않는다.

[중요 규칙]
- 반드시 한국어로 작성한다.
- 출력 안에서 [2], [3] 대제목은 그대로 사용하라.
- (1) 주제별 기사 링크는 이 출력에 포함하지 말 것.

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
# 5) HTML 리포트 생성 (로컬에서 크게 보는 용도)
# ----------------------------------------
def build_html_report(
    mode_label: str,
    topic_map: Dict[str, List[int]],
    summary_items: List[Dict],
    analysis_body: str,
    out_dir: str,
) -> str:
    """
    - 텔레그램 텍스트와 동일한 내용을 조금 더 큰 화면에서 보기 위한 간단 HTML 리포트 생성.
    - out_dir 안에 'final_report.html'을 생성하고 해당 경로를 반환.
    """
    idx_map = {item["index"]: item for item in summary_items}

    html_parts: List[str] = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html lang='ko'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8' />")
    html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1' />")
    html_parts.append("<title>신문 1면 분석 리포트</title>")
    html_parts.append(
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
        "padding:16px;max-width:960px;margin:0 auto;line-height:1.6;}"
        "h1{font-size:1.6rem;margin-bottom:0.5rem;}"
        "h2{font-size:1.2rem;margin-top:1.4rem;border-bottom:1px solid #ddd;padding-bottom:0.2rem;}"
        "code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:12px;border-radius:8px;}"
        "section{margin-bottom:1.5rem;}"
        "ul{padding-left:1.2rem;}"
        "a{color:#0366d6;text-decoration:none;}"
        "a:hover{text-decoration:underline;}"
        "</style>"
    )
    html_parts.append("</head>")
    html_parts.append("<body>")

    html_parts.append("<header>")
    html_parts.append("<h1>신문 1면 분석 리포트</h1>")
    html_parts.append(f"<p><strong>모드:</strong> {html.escape(mode_label)}</p>")
    html_parts.append("</header>")

    # 주제별 기사 링크
    html_parts.append("<section>")
    html_parts.append("<h2>주제별 기사 링크</h2>")

    if topic_map:
        for topic_label in sorted(topic_map.keys()):
            html_parts.append(f"<h3>{html.escape(topic_label)}</h3>")
            html_parts.append("<ul>")
            for idx in topic_map[topic_label]:
                item = idx_map.get(idx)
                if not item:
                    continue
                source = item.get("source", "언론사 미상")
                url = item.get("url", "")
                html_parts.append(
                    f"<li>{html.escape(str(source))}: "
                    f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>"
                    f"{html.escape(url)}</a></li>"
                )
            html_parts.append("</ul>")
    else:
        html_parts.append("<p>주제별 분류에 실패하여 단순 링크 목록으로 대체되었습니다.</p>")
        html_parts.append("<ul>")
        for item in summary_items:
            idx = item["index"]
            source = item["source"]
            url = item["url"]
            html_parts.append(
                f"<li>[{idx}] {html.escape(str(source))}: "
                f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>"
                f"{html.escape(url)}</a></li>"
            )
        html_parts.append("</ul>")

    html_parts.append("</section>")

    # 분석 본문
    html_parts.append("<section>")
    html_parts.append("<h2>분석</h2>")
    escaped = html.escape(analysis_body)
    html_parts.append(f"<pre>{escaped}</pre>")
    html_parts.append("</section>")

    html_parts.append("</body></html>")

    html_text = "\n".join(html_parts)
    html_path = os.path.join(out_dir, "final_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    log(f"[단계] HTML 리포트 파일 저장: {html_path}")
    return html_path


# ----------------------------------------
# 6) 텔레그램으로 결과 전송
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
# 7) 전체 파이프라인
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

    # 링크 목록 생성 (주제별) — 리포트 최상단으로 이동
    link_lines: List[str] = []
    if topic_map:
        link_lines.append("[주제별 기사 링크]")
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

    # 최종 텍스트 리포트:
    # 1) [모드] 헤더
    # 2) 주제별 기사 링크
    # 3) 구분선
    # 4) (2) 주제별 핵심 요약, (3) 언론사별 중요 포인트, (4)(5) 전체 해석/비판 포인트
    final_report = header + "\n" + links_section + "\n\n----------\n" + analysis_body

    # 최종 리포트 텍스트 파일 저장
    report_path = os.path.join(out_dir, "final_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    log(f"[단계] 최종 리포트 파일 저장: {report_path}")

    # HTML 리포트 파일도 함께 생성 (로컬 브라우저에서 읽기 좋게)
    build_html_report(mode_label, topic_map, summary_items, analysis_body, out_dir)

    # 텔레그램으로도 전송
    send_telegram_message(final_report)

    total_dt = time.perf_counter() - start_all
    log(f"[완료] 전체 파이프라인 종료 (총 소요 {total_dt:.1f}초)")


if __name__ == "__main__":
    run_pipeline()
