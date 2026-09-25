"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://global.toyota/en/newsroom/corporate/39241625.html",
    "https://global.toyota/en/newsroom/corporate/32120629.html",
    "https://global.toyota/en/newsroom/corporate/41363981.html",
    "https://global.toyota/en/newsroom/corporate/40554519.html",
    "https://global.toyota/en/newsroom/corporate/42267575.html",
]


async def crawl_article(url: str) -> dict:
    """Crawl one public article and return the required landing metadata."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    if not result.success:
        raise RuntimeError(result.error_message or "Crawler returned an unsuccessful result")

    metadata = result.metadata or {}
    content_markdown = str(result.markdown).strip()
    if not content_markdown:
        raise ValueError("Crawler returned empty article content")

    return {
        "url": url,
        "title": metadata.get("title") or "Unknown",
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())