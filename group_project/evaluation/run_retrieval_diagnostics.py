"""Measure paired A/B retrieval locally, without sending corpus text to APIs."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve

from .run_ab_evaluation import (
    DEFAULT_EMBEDDING_MODEL,
    EVAL_THRESHOLD,
    HERE,
    TOP_K,
    build_snapshot,
    load_cases,
)


OUTPUT = HERE / "retrieval_diagnostics.json"


def evidence_coverage(evidence: str, contexts: list[str]) -> float:
    """Highest token coverage in one chunk; a retrieval proxy, not a RAGAS score."""
    terms = set(re.findall(r"\w+", evidence.casefold()))
    if not terms or not contexts:
        return 0.0
    return max(
        len(terms & set(re.findall(r"\w+", context.casefold()))) / len(terms)
        for context in contexts
    )


def inspect_case(case: dict, config: str) -> dict:
    started = perf_counter()
    results = (
        semantic_search(case["question"], top_k=TOP_K)
        if config == "A" else retrieve(
            case["question"], top_k=TOP_K,
            score_threshold=EVAL_THRESHOLD, use_reranking=True,
        )
    )
    source = Path(case["source"]).name
    evidence = " ".join(case["expected_context"].split())
    contexts = [result["content"] for result in results]
    source_ranks = [
        rank for rank, result in enumerate(results, 1)
        if result["metadata"].get("source") == source
    ]
    evidence_ranks = [
        rank for rank, result in enumerate(results, 1)
        if result["metadata"].get("source") == source
        and evidence in " ".join(result["content"].split())
    ]
    return {
        "id": case["id"],
        "category": case.get("category", "policy"),
        "question": case["question"],
        "config": config,
        "expected_source": source,
        "expected_context": case["expected_context"],
        "source_rank": source_ranks[0] if source_ranks else None,
        "evidence_rank": evidence_ranks[0] if evidence_ranks else None,
        "evidence_coverage": round(evidence_coverage(evidence, contexts), 3),
        "source_ids": [result["id"] for result in results],
        "source_paths": [result["metadata"].get("source") for result in results],
        "retrieved_contexts": contexts,
        "latency_seconds": round(perf_counter() - started, 3),
    }


def main() -> None:
    from os import environ
    from src.task4_chunking_indexing import chunk_documents, load_documents

    environ["EMBEDDING_MODEL"] = environ.get("EVALUATION_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    cases = load_cases()
    snapshot = build_snapshot()
    indexed_chunks = chunk_documents(load_documents())
    rows = []
    for index, case in enumerate(cases, 1):
        source = Path(case["source"]).name
        evidence = " ".join(case["expected_context"].split())
        indexed = any(
            chunk["metadata"].get("source") == source
            and evidence in " ".join(chunk["content"].split())
            for chunk in indexed_chunks
        )
        for config in ("A", "B"):
            row = inspect_case(case, config)
            row["gold_evidence_indexed"] = indexed
            rows.append(row)
        print(f"Inspected {index}/{len(cases)} golden questions", flush=True)
    summary = {
        config: {
            "expected_source_hit": sum(
                row["source_rank"] is not None for row in rows if row["config"] == config
            ),
            "exact_evidence_hit": sum(
                row["evidence_rank"] is not None for row in rows if row["config"] == config
            ),
        }
        for config in ("A", "B")
    }
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": environ["EMBEDDING_MODEL"],
        "top_k": TOP_K,
        "score_threshold": EVAL_THRESHOLD,
        "pageindex_enabled": False,
        "case_count": len(cases),
        **snapshot,
        "summary": summary,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
