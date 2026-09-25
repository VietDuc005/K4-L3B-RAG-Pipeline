"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import os
from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

load_dotenv()

# Ngưỡng cosine similarity mặc định (có thể đọc từ .env nếu đã hiệu chỉnh)
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3") or "0.3")
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    if top_k <= 0 or not query.strip():
        return []

    # 1. Lấy kết quả từ dense và lexical search
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)

    # 2. Fuse hai danh sách bằng RRF đúng một lần (nếu use_reranking=True)
    if use_reranking:
        hybrid = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        hybrid = dense[:top_k]

    # 3. Lấy best cosine score gốc từ dense search
    best_dense_score = float(dense[0]["score"]) if dense else 0.0

    # 4. Nếu best_dense_score < score_threshold, thử PageIndex fallback
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            pass

    # 5. Nếu dense score đủ hoặc fallback không có kết quả, trả hybrid results
    return hybrid[:top_k]


if __name__ == "__main__":
    test_queries = [
        ("What is Toyota's anti-bribery policy?", "In-domain query"),
        ("How to bake chocolate cookies?", "Out-of-domain query"),
    ]

    print("=== Testing Retrieval Pipeline & Calibration ===")
    for q, desc in test_queries:
        print(f"\n[{desc}] Query: '{q}'")
        dense = semantic_search(q, top_k=3)
        dense_score = dense[0]["score"] if dense else 0.0
        print(f"-> Best Dense Cosine Score: {dense_score:.4f}")
        res = retrieve(q, top_k=3, score_threshold=0.35)
        if res:
            top_hit = res[0]
            print(f"Top 1 Hit ID: {top_hit['id']}")
            print(f"Score: {top_hit['score']:.6f} | Method: {top_hit['retrieval_method']}")
            print(f"Title: {top_hit['metadata'].get('title')}")
            print(f"Excerpt: {top_hit['content'][:120]}...")
        else:
            print("No hits found (fallback safe).")
