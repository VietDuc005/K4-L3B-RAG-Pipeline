"""
Task 7 — Reciprocal Rank Fusion (RRF).

RRF gộp nhiều bảng xếp hạng (Dense và BM25) mà không cộng trực tiếp điểm số khác thang đo.
Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.
RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.
"""

from typing import Any


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult theo thứ tự score RRF giảm dần."""
    if top_k <= 0 or not ranked_lists:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        seen_in_current_list: set[str] = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = str(item["id"])
            if item_id in seen_in_current_list:
                continue
            seen_in_current_list.add(item_id)

            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
            if item_id not in items:
                # Copy item để không gây mutation lên danh sách ban đầu
                items[item_id] = dict(item)

    # Sắp xếp giảm dần theo điểm RRF
    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    results = []
    for item_id in sorted_ids[:top_k]:
        # Copy item trước khi cập nhật score và retrieval_method
        fused_item = dict(items[item_id])
        fused_item["score"] = float(scores[item_id])
        fused_item["retrieval_method"] = "hybrid"
        results.append(fused_item)

    return results


if __name__ == "__main__":
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search

    query = "What is Toyota's anti-bribery policy?"
    dense_results = semantic_search(query, top_k=5)
    bm25_results = lexical_search(query, top_k=5)

    print(f"Dense top 1: {dense_results[0]['id']} (score: {dense_results[0]['score']:.4f})")
    print(f"BM25 top 1:  {bm25_results[0]['id']} (score: {bm25_results[0]['score']:.4f})")

    hybrid_results = rerank_rrf([dense_results, bm25_results], top_k=5)
    print("\n--- Fused Hybrid Results (RRF) ---")
    for idx, hit in enumerate(hybrid_results, 1):
        print(f"[{idx}] Score: {hit['score']:.6f} | ID: {hit['id']} | Method: {hit['retrieval_method']}")
        print(f"    Title: {hit['metadata']['title']}")
        print(f"    Excerpt: {hit['content'][:100]}...\n")
