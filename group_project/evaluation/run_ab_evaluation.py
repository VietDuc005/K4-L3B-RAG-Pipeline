"""Run a paired dense-vs-hybrid RAGAS evaluation on an exact corpus snapshot."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from dotenv import load_dotenv

from src.task10_generation import (
    DEFAULT_MODELS,
    PROVIDER_KEYS,
    SAFE_REFUSAL,
    generate_from_sources,
)
from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASET = HERE / "golden_dataset.json"
OUTPUT = HERE / "ab_results.json"
REPORT = HERE / "RESULT.md"
TOP_K = 5
EVAL_THRESHOLD = -1.0
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRICS = ("faithfulness", "answer_relevance", "context_recall", "context_precision")


def check_configuration(
    generator_provider: str | None = None,
    evaluator_provider: str | None = None,
    embedding_model: str | None = None,
    evaluator_model: str | None = None,
) -> dict:
    """Resolve and validate the experiment settings before API calls."""
    load_dotenv()
    generator_provider = (
        generator_provider or os.getenv("LLM_PROVIDER", "openai")
    ).strip().lower()
    evaluator_provider = (
        evaluator_provider or os.getenv("EVALUATOR_PROVIDER", "openai")
    ).strip().lower()
    if generator_provider not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported generator provider: {generator_provider}")
    if evaluator_provider not in {"openai", "gemini"}:
        raise ValueError(f"Unsupported evaluator provider: {evaluator_provider}")
    for provider in {generator_provider, evaluator_provider}:
        key_name = PROVIDER_KEYS[provider]
        if not os.getenv(key_name, "").strip():
            raise RuntimeError(f"Set {key_name} in .env before running the benchmark")

    # call_llm reads these at call time; this override is local to the benchmark.
    model = os.getenv("LLM_MODEL", "").strip() if generator_provider == os.getenv("LLM_PROVIDER", "openai").strip().lower() else ""
    generator_model = model or DEFAULT_MODELS[generator_provider]
    os.environ["LLM_PROVIDER"] = generator_provider
    os.environ["LLM_MODEL"] = generator_model
    saved_evaluator = os.getenv("EVALUATOR_PROVIDER", "openai").strip().lower()
    saved_model = os.getenv("EVALUATOR_MODEL", "").strip() if evaluator_provider == saved_evaluator else ""
    evaluator_model = evaluator_model or saved_model or (
        "gemini-3.5-flash-lite" if evaluator_provider == "gemini" else "gpt-4o-mini"
    )
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    os.environ["EMBEDDING_MODEL"] = embedding_model
    return {
        "generator_provider": generator_provider,
        "generator_model": generator_model,
        "evaluator_provider": evaluator_provider,
        "evaluator_model": evaluator_model,
        "embedding_model": embedding_model,
        "evaluator_embedding_model": (
            "gemini-embedding-001" if evaluator_provider == "gemini"
            else "text-embedding-3-small"
        ),
    }


def load_cases() -> list[dict]:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) < 15:
        raise ValueError("Golden dataset must contain at least 15 cases")
    ids = set()
    for case in cases:
        if not all(case.get(key) for key in ("id", "question", "expected_answer", "expected_context")):
            raise ValueError("Every case needs id, question, expected_answer and expected_context")
        if case["id"] in ids:
            raise ValueError(f"Duplicate golden ID: {case['id']}")
        ids.add(case["id"])
        source = (ROOT / case["source"]).resolve()
        if not source.is_relative_to(ROOT / "data" / "standardized") or not source.is_file():
            raise ValueError(f"Golden source is not a standardized corpus file: {case['id']}")
        normalize = lambda value: " ".join(value.split())
        if normalize(case["expected_context"]) not in normalize(source.read_text(encoding="utf-8")):
            raise ValueError(f"Golden evidence not found in corpus: {case['id']}")
    return cases


def create_metrics(settings: dict) -> dict:
    """Use the RAGAS 0.4 collections API with one shared judge and embedder."""
    from openai import AsyncOpenAI
    from ragas.embeddings import GoogleEmbeddings
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    if settings["evaluator_provider"] == "gemini":
        from google import genai

        key = os.environ["GEMINI_API_KEY"]
        client = AsyncOpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        judge = llm_factory(settings["evaluator_model"], client=client)
        # Keep this client alive: GoogleEmbeddings makes its own API requests.
        embedding_client = genai.Client(api_key=key)
        embeddings = GoogleEmbeddings(
            client=embedding_client, model=settings["evaluator_embedding_model"]
        )
    else:
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        judge = llm_factory(settings["evaluator_model"], client=client)
        embeddings = embedding_factory(
            "openai", model=settings["evaluator_embedding_model"], client=client
        )
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevance": AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=1),
        "context_recall": ContextRecall(llm=judge),
        "context_precision": ContextPrecisionWithReference(llm=judge),
    }


def build_snapshot() -> dict:
    """Index only the current standardized corpus in memory for this run."""
    # The model is downloaded once before benchmarking. Offline mode avoids
    # a network HEAD request on every rerun and keeps the snapshot reproducible.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import chromadb

    from src import task5_semantic_search as semantic_module
    from src.task4_chunking_indexing import chunk_documents, embed_texts, load_documents

    documents = load_documents()
    chunks = chunk_documents(documents)
    if not chunks:
        raise RuntimeError("No standardized documents to evaluate")
    collection = chromadb.EphemeralClient().get_or_create_collection(
        "evaluation_documents", metadata={"hnsw:space": "cosine"}
    )
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        vectors = embed_texts([chunk["content"] for chunk in batch])
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=vectors,
            metadatas=[
                {key: value for key, value in chunk["metadata"].items() if value is not None}
                for chunk in batch
            ],
        )
        print(f"Indexed {min(start + 32, len(chunks))}/{len(chunks)} chunks", flush=True)
    # semantic_search imported get_collection directly; replace that alias only
    # inside this benchmark process, leaving the tracked Chroma DB untouched.
    semantic_module.get_collection = lambda: collection

    fingerprint = hashlib.sha256()
    for path in sorted((ROOT / "data" / "standardized").rglob("*.md")):
        fingerprint.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        fingerprint.update(path.read_bytes())
    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "corpus_sha256": fingerprint.hexdigest(),
        "corpus_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "golden_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
    }


async def score_case(metrics: dict, case: dict, answer: str, contexts: list[str]) -> dict:
    """Use zero for missing evidence/refusal; otherwise calculate four judge scores."""
    if not contexts or answer == SAFE_REFUSAL:
        return {name: 0.0 for name in METRICS}
    question = case["question"]
    reference = case["expected_answer"]
    calls = {
        "faithfulness": metrics["faithfulness"].ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        ),
        "answer_relevance": metrics["answer_relevance"].ascore(
            user_input=question, response=answer
        ),
        "context_recall": metrics["context_recall"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        "context_precision": metrics["context_precision"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
    }
    results = await asyncio.gather(*calls.values())
    scores = {name: float(result.value) for name, result in zip(calls, results)}
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError(f"RAGAS returned a non-finite score for {case['id']}")
    return scores


async def evaluate(cases: list[dict], metrics: dict) -> list[dict]:
    from src.task4_chunking_indexing import chunk_documents, load_documents

    indexed_chunks = chunk_documents(load_documents())
    rows = []
    for index, case in enumerate(cases, 1):
        expected_source = Path(case["source"]).name
        normalized_evidence = " ".join(case["expected_context"].split())
        gold_evidence_indexed = any(
            chunk["metadata"].get("source") == expected_source
            and normalized_evidence in " ".join(chunk["content"].split())
            for chunk in indexed_chunks
        )
        for config, use_reranking in (("A", False), ("B", True)):
            started = perf_counter()
            # Negative threshold disables PageIndex in both arms. The only
            # difference is dense-only versus dense + BM25 + one RRF merge.
            chunks = (
                retrieve(
                    case["question"], top_k=TOP_K,
                    score_threshold=EVAL_THRESHOLD, use_reranking=True,
                )
                if use_reranking else semantic_search(case["question"], top_k=TOP_K)
            )
            generated = generate_from_sources(
                case["question"], chunks, raise_on_error=True
            )
            latency_seconds = round(perf_counter() - started, 3)
            contexts = [item["content"] for item in chunks]
            scores = await score_case(metrics, case, generated["answer"], contexts)
            rows.append({
                "id": case["id"],
                "question": case["question"],
                "category": case.get("category", "policy"),
                "config": config,
                "answer": generated["answer"],
                "expected_answer": case["expected_answer"],
                "expected_context": case["expected_context"],
                "gold_evidence_indexed": gold_evidence_indexed,
                "source_ids": [item["id"] for item in chunks],
                "source_paths": [item["metadata"].get("source") for item in chunks],
                "expected_source": expected_source,
                "expected_source_hit": any(
                    item["metadata"].get("source") == expected_source for item in chunks
                ),
                "expected_evidence_hit": any(
                    item["metadata"].get("source") == expected_source
                    and normalized_evidence in " ".join(item["content"].split())
                    for item in chunks
                ),
                "retrieved_contexts": contexts,
                "context_chars": sum(len(text) for text in contexts),
                "latency_seconds": latency_seconds,
                "scores": scores,
            })
        print(f"Scored {index}/{len(cases)} golden questions")
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        config: {
            **{metric: mean(row["scores"][metric] for row in rows if row["config"] == config)
               for metric in METRICS},
            "latency_seconds": mean(row["latency_seconds"] for row in rows if row["config"] == config),
        }
        for config in ("A", "B")
    }


def failure_analysis(row: dict) -> tuple[str, str, str, str]:
    """Give an evidence-based first diagnosis for manual review of a case."""
    sources = ", ".join(row["source_paths"][:3]) or "none"
    if not row["expected_source_hit"]:
        return (
            "retrieval",
            f"Expected {row['expected_source']} is absent from top-{TOP_K}; top sources: {sources}.",
            "Remove corpus noise or revise retrieval ranking so the correct document enters top-k.",
            "Expected-source hit and context recall should increase on this case.",
        )
    if not row["expected_evidence_hit"]:
        if row.get("gold_evidence_indexed") is False:
            return (
                "data/chunking",
                f"Gold evidence from {row['expected_source']} does not survive as one indexed chunk.",
                "Repair source conversion or change chunk boundaries so the evidence remains intact.",
                "The evidence should become indexed, then appear in top-k on a rerun.",
            )
        return (
            "retrieval",
            f"{row['expected_source']} is in top-{TOP_K} and gold evidence is indexed, but its chunk is absent from top-{TOP_K}.",
            "Reduce noisy chunks and improve within-document ranking of evidence-bearing chunks.",
            "Expected-evidence hit and context recall should increase on this case.",
        )
    if row["answer"] == SAFE_REFUSAL:
        return (
            "generation",
            "Gold evidence is retrieved, but the generator returned a safe refusal.",
            "Review the refusal instruction and citation prompt using this retrieved context.",
            "The case should receive a cited answer with higher relevance and faithfulness.",
        )
    if row["scores"]["faithfulness"] < 0.5:
        return (
            "generation",
            "Gold evidence is retrieved, but the judge marked the answer as weakly grounded.",
            "Inspect unsupported claims and tighten citation-grounding instructions.",
            "Faithfulness should rise without reducing context recall.",
        )
    if row["scores"]["answer_relevance"] < 0.5:
        return (
            "generation",
            "Gold evidence is retrieved, but the answer does not directly address the question.",
            "Constrain the response to the requested fact and keep the relevant citation.",
            "Answer relevance should rise while faithfulness remains stable.",
        )
    return (
        "retrieval/generation",
        f"Gold evidence appears in top-{TOP_K}; inspect unrelated chunks and the answer for this low aggregate score.",
        "Review the per-metric scores and retrieved contexts before changing one stage.",
        "The case-level metric that is lowest should improve on a rerun.",
    )


def report_markdown(run: dict) -> str:
    summary = run["summary"]
    rows_by_config = {
        config: {row["id"]: row for row in run["rows"] if row["config"] == config}
        for config in ("A", "B")
    }
    mean_score = lambda row: mean(row["scores"].values())
    worst = sorted(rows_by_config["B"].values(), key=mean_score)[:3]
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    a_avg = mean(summary["A"][metric] for metric in METRICS)
    b_avg = mean(summary["B"][metric] for metric in METRICS)
    improved = [metric for metric in METRICS if summary["B"][metric] > summary["A"][metric] + 0.01]
    declined = [metric for metric in METRICS if summary["B"][metric] < summary["A"][metric] - 0.01]
    lost_evidence = [
        case_id for case_id, a in rows_by_config["A"].items()
        if a["expected_evidence_hit"] and not rows_by_config["B"][case_id]["expected_evidence_hit"]
    ]
    mean_chars = {
        config: mean(row["context_chars"] for row in run["rows"] if row["config"] == config)
        for config in ("A", "B")
    }
    lines = [
        "# RAG evaluation results",
        "",
        "## Run information",
        "",
        f"- Date (UTC): {run['timestamp']}",
        f"- Dataset: {run['case_count']} cases; SHA-256 `{run['golden_sha256']}`",
        f"- Framework: RAGAS {run['ragas_version']}",
        f"- Generator: {run['generator_provider']} / {run['generator_model']}",
        f"- Evaluator: {run['evaluator_provider']} / {run['evaluator_model']}; answer-relevance embedding `{run['evaluator_embedding_model']}`; strictness 1",
        f"- Corpus commit: `{run['corpus_commit']}`; exact standardized-file SHA-256 `{run['corpus_sha256']}`",
        f"- Corpus snapshot: {run['document_count']} documents, {run['chunk_count']} chunks; dense embedding `{run['embedding_model']}`",
        f"- `top_k`: {TOP_K}; actual score threshold: {EVAL_THRESHOLD} (PageIndex disabled in both configurations)",
        "- Corpus index was rebuilt in memory for this run; no stale tracked Chroma chunks were used.",
        "- Raw answers, contexts and per-case scores: `ab_results.json`",
        "",
        "## Configurations",
        "",
        "- Config A: dense-only retrieval.",
        "- Config B: dense + BM25, merged once using RRF.",
        "- Both use the same corpus, golden cases, generator, evaluator, prompt and `top_k`.",
        "- Retrieval and generation latency excludes evaluator calls and index build time.",
        "",
        "## Overall scores",
        "",
        "| Metric | Config A | Config B | Delta B−A |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer relevance",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
    }
    for metric in METRICS:
        a, b = summary["A"][metric], summary["B"][metric]
        lines.append(f"| {labels[metric]} | {a:.3f} | {b:.3f} | {b-a:+.3f} |")
    lines.extend([
        f"| Average | {a_avg:.3f} | {b_avg:.3f} | {b_avg-a_avg:+.3f} |",
        "",
        "## A/B comparison",
        "",
        f"- B improves: {', '.join(improved) if improved else 'none'}; "
        f"B declines: {', '.join(declined) if declined else 'none'} (material delta > 0.01).",
        f"- Average delta B−A: {b_avg-a_avg:+.3f}. "
        f"Cases losing the exact gold evidence under B: {', '.join(lost_evidence) if lost_evidence else 'none'}.",
        f"- Mean end-to-end latency: A {summary['A']['latency_seconds']:.2f}s; "
        f"B {summary['B']['latency_seconds']:.2f}s per question.",
        f"- Mean retrieved context size: A {mean_chars['A']:.0f} characters; "
        f"B {mean_chars['B']:.0f} characters. Token usage and monetary cost were not collected.",
        "- Do not choose a winner from the average alone; inspect lost-evidence and worst cases below.",
        "- Refusals and missing contexts receive zero for all four metrics.",
        "",
        "## Worst performers",
        "",
        "The three lowest-scoring unique cases under Config B are shown with the paired A result.",
        "Stage and cause below are diagnostic hypotheses tied to retrieved evidence; read the raw case trace before changing the pipeline.",
        "",
        "| Case | A mean | B mean | B source/evidence hit | Failure stage | Root cause evidence |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ])
    for row in worst:
        stage, evidence, _, _ = failure_analysis(row)
        paired = rows_by_config["A"][row["id"]]
        lines.append(
            f"| {row['id']} | {mean_score(paired):.3f} | {mean_score(row):.3f} | "
            f"{row['expected_source_hit']}/{row['expected_evidence_hit']} | "
            f"{stage} | {clean(evidence)} |"
        )
    lines.append("")
    for row in worst:
        lines.extend([
            f"**{row['id']} — {clean(row['question'])}**",
            f"Expected: {clean(row['expected_answer'])}",
            f"Config B answer: {clean(row['answer'])}",
            f"Config B metrics: " + ", ".join(
                f"{metric}={row['scores'][metric]:.3f}" for metric in METRICS
            ) + ".",
            "",
        ])
    lines.extend([
        "## Recommendations",
        "",
        "| Priority | Action | Observed evidence | Expected impact | How to verify |",
        "| ---: | --- | --- | --- | --- |",
    ])
    for priority, row in enumerate(worst, 1):
        _, evidence, action, impact = failure_analysis(row)
        lines.append(
            f"| {priority} | {clean(action)} | {row['id']}: {clean(evidence)} | "
            f"{clean(impact)} | Rerun the same golden set and compare this case's four "
            f"metrics, evidence hit and latency against `ab_results.json`. |"
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "The judge scores are signals for case review, not ground truth. The golden set covers "
        "five policies and three newsroom cases; it does not yet include explicit out-of-domain refusals. "
        "The cost comparison is limited to context-size proxy because billed tokens were not captured. "
        "PageIndex fallback is outside this A/B experiment.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-provider", choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--evaluator-provider", choices=["openai", "gemini"])
    parser.add_argument("--evaluator-model")
    parser.add_argument("--embedding-model")
    args = parser.parse_args()
    settings = check_configuration(
        args.generator_provider, args.evaluator_provider,
        args.embedding_model, args.evaluator_model,
    )
    cases = load_cases()
    snapshot = build_snapshot()
    metrics = create_metrics(settings)
    rows = asyncio.run(evaluate(cases, metrics))
    from importlib.metadata import version

    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "ragas_version": version("ragas"),
        **settings,
        **snapshot,
        "top_k": TOP_K,
        "score_threshold": EVAL_THRESHOLD,
        "pageindex_enabled": False,
        "generation_temperature": 0.3,
        "summary": summarize(rows),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(report_markdown(run), encoding="utf-8")
    print(f"Wrote {OUTPUT} and {REPORT}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
