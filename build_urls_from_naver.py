import argparse
from typing import List, Tuple
from urllib.parse import urljoin
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

# Naver 신문사별 코드
PRESS_LIST: List[Tuple[str, str]] = [
    ("동아일보", "020"),
    ("한국일보", "469"),
    ("조선일보", "023"),
    ("중앙일보", "025"),
    ("한겨레", "028"),
    ("경향신문", "032"),
]

BASE_NEWPAPER_URL = "https://media.naver.com/press/{press}/newspaper?date={date}"


def get_kst_today() -> str:
    """현재 KST(UTC+9) 기준 날짜를 YYYYMMDD로 반환"""
    now_utc = datetime.utcnow()
    now_kst = now_utc + timedelta(hours=9)
    return now_kst.strftime("%Y%m%d")


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text


def extract_a1_links(html: str, page_url: str, press_code: str, date: str) -> List[str]:
    """
    newspaper 페이지에서 A1(1면) 기사 링크만 추출 시도.
    1) /article/newspaper/{press_code}/ & date={date} 포함 링크들 중에서
    2) 부모 텍스트에 'A1면', '1면' 등 키워드 있는지 확인
    3) 없으면 fallback으로 상위 기사 몇 개 사용
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if f"/article/newspaper/{press_code}/" not in href:
            continue
        if f"date={date}" not in href:
            continue

        full_url = urljoin(page_url, href)

        # 부모 쪽에 A1/1면 표시 있는지 확인
        is_a1 = False
        parent = a
        for _ in range(6):  # 위로 몇 단계만 올라가서 텍스트 검사
            parent = parent.parent
            if parent is None:
                break
            text = parent.get_text(" ", strip=True)
            if any(
                key in text
                for key in [
                    "A1면",
                    "A01면",
                    "A 1면",
                    "A 01면",
                    "1면",
                    "1 面",
                ]
            ):
                is_a1 = True
                break

        if is_a1:
            candidates.append(full_url)

    # A1 키워드 못 찾으면 fallback
    if not candidates:
        print(f"  [주의] A1면 키워드를 찾지 못해 {press_code}는 fallback으로 상위 기사 몇 개만 사용합니다.")
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/article/newspaper/{press_code}/" in href and f"date={date}" in href:
                full_url = urljoin(page_url, href)
                if full_url not in seen:
                    candidates.append(full_url)
                    seen.add(full_url)
            if len(candidates) >= 4:  # A1에 보통 3~4개 정도라고 가정
                break

    # 중복 제거
    unique_links: List[str] = []
    seen_urls = set()
    for u in candidates:
        if u not in seen_urls:
            unique_links.append(u)
            seen_urls.add(u)

    return unique_links


def build_urls_txt(date: str, output_path: str = "urls.txt") -> None:
    lines: List[str] = []

    print(f"[INFO] 사용할 신문 날짜(KST 기준): {date}")

    for press_name, press_code in PRESS_LIST:
        page_url = BASE_NEWPAPER_URL.format(press=press_code, date=date)
        print(f"[+] {press_name} ({press_code}) newspaper 페이지 크롤링: {page_url}")

        try:
            html = fetch_html(page_url)
        except Exception as e:
            print(f"    [에러] {press_name} 페이지 요청 실패: {e}")
            continue

        try:
            links = extract_a1_links(html, page_url, press_code, date)
        except Exception as e:
            print(f"    [에러] {press_name} A1 링크 추출 실패: {e}")
            continue

        if not links:
            print(f"    [주의] {press_name} A1 후보 링크 없음")
            continue

        print(f"    → {press_name} A1 추정 기사 {len(links)}개")

        lines.append(f"# {press_name}")
        lines.extend(links)
        lines.append("")

    if not lines:
        raise SystemExit("어떤 언론사에서도 링크를 추출하지 못했습니다. date나 HTML 구조를 확인하세요.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[완료] A1 링크 목록을 {output_path} 에 저장했습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="Naver 신문사 newspaper 페이지에서 A1(1면) 기사 링크 추출해 urls.txt 생성"
    )
    parser.add_argument(
        "--date",
        help="신문 발행일 (예: 20251117). 생략하면 오늘(KST) 기준으로 자동 계산.",
    )
    parser.add_argument(
        "--output",
        default="urls.txt",
        help="출력 파일명 (기본: urls.txt)",
    )
    args = parser.parse_args()

    date = args.date or get_kst_today()
    build_urls_txt(date=date, output_path=args.output)


if __name__ == "__main__":
    main()
