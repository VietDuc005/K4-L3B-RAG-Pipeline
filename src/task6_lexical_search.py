"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


import math
import numpy as np
from rank_bm25 import BM25Okapi

CORPUS: list[dict] = []


class CustomBM25(BM25Okapi):
    def _calc_idf(self, nd):
        """Standard BM25 IDF with a positive floor to avoid 0.0 IDF on small corpora."""
        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[word] = max(idf, 0.1)


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return CustomBM25(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global CORPUS
    corpus_to_use = CORPUS
    if not corpus_to_use:
        try:
            from src.task4_chunking_indexing import chunk_documents, load_documents

            documents = load_documents()
            CORPUS = chunk_documents(documents)
            corpus_to_use = CORPUS
        except Exception:
            corpus_to_use = []

    if not corpus_to_use:
        return []

    bm25 = build_bm25_index(corpus_to_use)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    indices = np.argsort(scores)[::-1]
    results = []
    for idx in indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        item = corpus_to_use[idx]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
        if len(results) >= top_k:
            break

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]



if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)

