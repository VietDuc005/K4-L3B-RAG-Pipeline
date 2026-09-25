# A/B evaluation: dense-only versus hybrid + RRF

## Run information and scope

This is a **local retrieval diagnostic, not a completed RAGAS A/B score**. The
four requested LLM-based metrics require sending retrieved corpus excerpts to an
external evaluator; that step has not been authorized. `retrieval_diagnostics.json`
contains the complete per-case top-5 traces from the local run. No metric
scores are invented below.

| Field | Value |
| --- | --- |
| Retrieval run | 2026-09-25 05:34:45 UTC |
| Corpus version | Git HEAD `00e0c435c96ec792c9804a1d582f6afeb1696867` plus local standardized files |
| Exact corpus SHA-256 | `b8f7fac36e5e768298e1a79439449a5fe0b711181f54949e295e2e1f34f38780` |
| Corpus snapshot | 10 Markdown documents; 611 chunks rebuilt in an in-memory Chroma collection |
| Golden dataset | 18 cases; SHA-256 `20895208f5294c03baff9b38fd284610cbda9c9a038c6ad56b842b8705a294e3` |
| Dense embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Planned generator / evaluator | Gemini `gemini-3.5-flash-lite` for both; not run on corpus |
| Planned evaluator embedding | Gemini `gemini-embedding-001`; not run on corpus |
| Shared `top_k` | 5 |
| Shared score threshold | `-1.0`, disabling PageIndex fallback in both arms |
| Chunking | Recursive, 500 characters, 50-character overlap |

The five policy documents supply 15 cases and three newsroom documents supply
three cases. Every `expected_context` is an excerpt checked against its
standardized Markdown source. The set includes exact-keyword policy questions,
semantic questions, and source-confusion cases. It does not test out-of-domain
refusal behavior. The Markdown describes collected Toyota publications; this
benchmark does not independently verify their current legal or factual status.

## A/B comparison

- **A, dense-only:** semantic top-5 retrieval.
- **B, hybrid + RRF:** dense top-10 and BM25 top-10 candidates fused once with
  reciprocal rank fusion, then truncated to top-5.

The golden cases, 611-chunk corpus snapshot, dense embedding, `top_k`, and
threshold were the same in both arms. The prepared scored runner also holds the
generator, evaluator, generation prompt, and generation settings constant.
Only the retrieval strategy changes. Index build time and model calls are
excluded from the local latency measurement.

## Overall scores

`N/A` means **not measured**, not zero. These are the required RAGAS metrics;
the retrieval hit counts below are separate diagnostics and must not be
substituted for them.

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | N/A | N/A | N/A |
| Answer relevance | N/A | N/A | N/A |
| Context recall | N/A | N/A | N/A |
| Context precision | N/A | N/A | N/A |

| Local retrieval diagnostic, 18 cases | A | B | Delta B−A |
| --- | ---: | ---: | ---: |
| Expected document in top-5 | 18/18 | 18/18 | 0 |
| Exact gold span **from expected document** in top-5 | 13/18 | 15/18 | +2 cases |
| Mean retrieval latency | 0.072 s | 0.150 s | +0.077 s |
| Mean retrieved context size | 2,054 chars | 2,101 chars | +47 chars |

The hit check requires both the expected document and the exact gold span in
one selected chunk. This prevents a sentence copied into another policy from
being counted as evidence for the target policy. These timings are local
single-process observations, not a cost estimate or stable production latency
benchmark. Token use and monetary cost have not been measured.

## Worst performers (retrieval diagnostic)

These are the three cases where **B still fails the exact, source-aware evidence
check**. They are not ranked by RAGAS, because no answer or judge score exists.
In all three, the gold passage survives as one indexed chunk, but neither arm
places it in top-5. That makes retrieval ranking the observed failure stage;
generation has not been tested for these cases.

| Case | A/B gold span rank | Observed root cause | Failure stage |
| --- | --- | --- | --- |
| `case_01`, anti-bribery policy scope | absent / absent | The correct passage is indexed as `legal/toyota_global_abac_policy.md::chunk-8`. B's top-5 includes the *same sentence* from the separate Speak Up policy at rank 4, while anti-bribery footer/header chunks occupy high ranks. A correct-looking answer could cite the wrong policy. | Retrieval: source confusion and noisy ranking |
| `case_16`, 2023 cloud incident exclusions | absent / absent | The exclusion statement is indexed as `news/article_01.md::chunk-62`. B ranks the page's search/navigation heading first and other incident table fragments above the explicit exclusion. All five returned chunks come from the right article, so document-level hit alone hides the missing answer. | Retrieval: within-document ranking and page noise |
| `case_18`, certification decision owner | absent / absent | The CTO versus G-CQO distinction is indexed as `news/article_03.md::chunk-64`. B selects the related responsibility/process chunk 65 at rank 3, plus article heading and surrounding sections, but omits chunk 64. A similarly misses the direct role assignment. | Retrieval: adjacent-chunk ranking |

The two cases gained by B are `case_02` (facilitation payments, B rank 1)
and `case_08` (human-rights remedy, B rank 4); both lack the exact evidence in
A's top-5. B loses no source-aware gold span found by A. This is evidence of a
retrieval gain on this dataset, while the roughly doubled local retrieval time
and three remaining misses prevent a blanket recommendation to deploy B.

## Recommendations and verification

| Priority | Change tied to observed failure | Expected impact | Verification |
| ---: | --- | --- | --- |
| 1 | Strip repeated PDF headers/footers and website navigation before chunking. `case_01` ranks a policy footer; `case_16` ranks a search heading. | Less irrelevant context and improved context precision; the gold chunks may rise into top-5. | Rebuild the same 10-document snapshot, rerun the 18-case diagnostic, inspect gold ranks for `case_01` and `case_16`, then compare all four RAGAS metrics and latency once authorized. |
| 2 | Test source-aware ranking or a second-stage chunk reranker on `case_01` and `case_18`, changing this retrieval component alone in a new run. | More exact evidence from the correct document; improved context recall without an unjustified cross-policy citation. | Compare source-aware gold-span hit on these two IDs and the full 18 cases against this run; inspect whether any previously successful case loses evidence, and record latency. |
| 3 | Run the paired generator + RAGAS benchmark once corpus transfer is approved, then read the three lowest *scored* cases. | Establish faithfulness, answer relevance, context recall, context precision, and B−A deltas; detect generation failures even where retrieval succeeds. | Run `python -m group_project.evaluation.run_ab_evaluation --generator-provider gemini --evaluator-provider gemini` with the same corpus/dataset hashes and `top_k=5`. Review `ab_results.json` and its regenerated report; investigate any case whose exact gold evidence is present but answer grounding or relevance is low. |

The OpenAI account returned `credit_balance_exhausted` in a small service test,
so the prepared runner can use Gemini for both generator and evaluator after
explicit authorization. A public synthetic test confirmed that the installed
RAGAS version (0.4.3) can call the Gemini judge. No real corpus was sent in
those checks. The recommendation to clean corpus text and the recommendation
to change ranking are **separate experiments**; applying both before a rerun
would obscure which change caused any metric delta.

## Reproduction

Run `python -m group_project.evaluation.run_retrieval_diagnostics` from the
repository root in the project virtual environment to regenerate the local
trace. Keep `golden_dataset.json` and `data/standardized` at the hashes above
for a direct comparison. The tracked persistent Chroma collection contains
older chunks, so both evaluation scripts build a fresh in-memory snapshot.
The scored runner has been prepared but must not be run with corpus content
until external model transfer is authorized. There are no bonus experiments.
