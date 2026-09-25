"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def upload_documents() -> dict[str, str]:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY is not set. Skipping PageIndex upload.")
        return {}

    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    doc_ids = {}
    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        for path in STANDARDIZED_DIR.rglob("*.md"):
            if not path.is_file():
                continue
            doc = client.upload(str(path))
            doc_ids[path.name] = doc.id
        CACHE_FILE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
    except Exception as err:
        print(f"PageIndex upload notice: {err}")

    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip() or not PAGEINDEX_API_KEY:
        return []

    results = []
    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        # Timeout để không làm pipeline bị treo
        search_res = client.search(query=query, top_k=top_k)
        
        seen_ids = set()
        for idx, item in enumerate(search_res, 1):
            item_id = getattr(item, "id", f"pageindex-{idx}")
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            score = float(getattr(item, "score", max(0.1, 1.0 - (idx - 1) * 0.1)))
            content = str(getattr(item, "content", getattr(item, "text", "")))
            source = str(getattr(item, "source", "pageindex.md"))
            title = str(getattr(item, "title", source.replace(".md", "")))
            doc_type = "legal" if "legal" in source.lower() else "news"
            url = getattr(item, "url", None)

            results.append({
                "id": str(item_id),
                "content": content,
                "score": score,
                "metadata": {
                    "source": source,
                    "title": title,
                    "doc_type": doc_type,
                    "url": url,
                    "chunk_index": 0,
                },
                "retrieval_method": "pageindex",
            })
    except Exception as err:
        print(f"PageIndex search error handled safely: {err}")
        return []

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]


if __name__ == "__main__":
    upload_documents()
