from __future__ import annotations

import argparse
import json
import time

import mlflow
from mlflow.entities import SpanType

from backend.app.config import ROOT, load_project_config
from pipelines.indexing import CorpusIndex, SemanticIndex, load_processed_papers
from pipelines.indexing.search import PROCESSED_PAPERS

RELEVANCE_FILE = ROOT / "data" / "evaluation" / "relevance.json"
TRACE_IDS_FILE = ROOT / ".cache" / "evidencemap" / "latest_evaluation_trace_ids.json"


def compute_retrieval_metrics(
    results: list[dict], relevant_paper_ids: set[str], top_k: int
) -> dict[str, float]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not relevant_paper_ids:
        raise ValueError("At least one relevant paper ID is required")

    retrieved_ids = [result["paper_id"] for result in results[:top_k]]
    relevant_ranks = [
        rank
        for rank, paper_id in enumerate(retrieved_ids, start=1)
        if paper_id in relevant_paper_ids
    ]
    return {
        "recall_at_k": len(set(retrieved_ids) & relevant_paper_ids) / len(relevant_paper_ids),
        "mrr_at_k": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
    }


def _write_trace_ids(trace_ids: list[str]) -> None:
    TRACE_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TRACE_IDS_FILE.with_suffix(".tmp")
    temp_path.write_text(json.dumps(trace_ids, indent=2), encoding="utf-8")
    temp_path.replace(TRACE_IDS_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate lexical and semantic retrieval against relevance labels."
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not RELEVANCE_FILE.exists():
        raise FileNotFoundError(f"Relevance labels not found: {RELEVANCE_FILE}")

    papers = load_processed_papers()
    if not papers:
        raise RuntimeError(f"No processed papers found at {PROCESSED_PAPERS}")

    cfg = load_project_config()
    labels = json.loads(RELEVANCE_FILE.read_text(encoding="utf-8"))
    paper_ids = {paper["paper_id"] for paper in papers}
    indexes = {
        "lexical": CorpusIndex(papers),
        "semantic": SemanticIndex(papers, cfg["models"]["embedding"]),
    }
    semantic_warmup_seconds = indexes["semantic"].warmup()
    print(f"semantic model warm-up: {semantic_warmup_seconds:.2f}s")

    for item in labels["queries"]:
        unknown_ids = set(item["relevant_paper_ids"]) - paper_ids
        if unknown_ids:
            raise ValueError(
                f"Unknown relevant paper IDs for {item['query']!r}: {sorted(unknown_ids)}"
            )

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    experiment = mlflow.set_experiment("evidencemap-retrieval-evaluation")
    trace_ids: list[str] = []
    TRACE_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACE_IDS_FILE.write_text("[]", encoding="utf-8")

    for item in labels["queries"]:
        query = item["query"]
        domain = item["domain"]
        relevant_ids = set(item["relevant_paper_ids"])
        for mode, index in indexes.items():
            with mlflow.start_run(run_name=f"{mode}: {query}"):
                with mlflow.start_span(
                    name="evaluate_retrieval_query", span_type=SpanType.CHAIN
                ) as root_span:
                    root_span.set_inputs({
                        "query": query,
                        "domain": domain,
                        "mode": mode,
                        "top_k": args.top_k,
                        "relevant_paper_ids": sorted(relevant_ids),
                    })
                    started = time.perf_counter()
                    with mlflow.start_span(
                        name=f"{mode}_retrieval", span_type=SpanType.RETRIEVER
                    ) as retrieval_span:
                        retrieval_span.set_inputs({
                            "query": query,
                            "domain": domain,
                            "top_k": args.top_k,
                        })
                        results = index.search(query, limit=args.top_k, domain=domain)
                        latency_ms = (time.perf_counter() - started) * 1000
                        retrieval_span.set_outputs({
                            "results": results,
                            "latency_ms": latency_ms,
                        })

                    with mlflow.start_span(
                        name="compute_relevance_metrics", span_type=SpanType.EVALUATOR
                    ) as evaluator_span:
                        evaluator_span.set_inputs({
                            "retrieved_paper_ids": [
                                result["paper_id"] for result in results
                            ],
                            "relevant_paper_ids": sorted(relevant_ids),
                            "top_k": args.top_k,
                        })
                        metrics = compute_retrieval_metrics(
                            results, relevant_ids, args.top_k
                        )
                        evaluator_span.set_outputs(metrics)

                    root_span.set_outputs({
                        "metrics": metrics,
                        "results": results,
                    })

                trace_id = mlflow.get_last_active_trace_id()
                if not trace_id:
                    raise RuntimeError(f"MLflow did not return a trace ID for {mode}: {query}")
                trace_ids.append(trace_id)
                _write_trace_ids(trace_ids)

                mlflow.log_params({
                    "mode": mode,
                    "query": query,
                    "domain": domain,
                    "top_k": args.top_k,
                    "corpus_size": len(papers),
                    "label_set_version": labels["version"],
                    "labeling_notes": labels["labeling_notes"][:250],
                    "timing_protocol": "warm_model",
                })
                mlflow.log_metrics({
                    **metrics,
                    "search_latency_ms": latency_ms,
                })
                if mode == "semantic":
                    mlflow.log_param("embedding_model", index.model_name)
                    mlflow.log_metric("model_warmup_seconds", semantic_warmup_seconds)
                    mlflow.log_metric("embedding_index_build_seconds", index.build_seconds)
                mlflow.log_dict({
                    "query": query,
                    "domain": domain,
                    "relevant_paper_ids": sorted(relevant_ids),
                    "results": results,
                    "metrics": metrics,
                    "labeling_notes": labels["labeling_notes"],
                }, "evaluation.json")

            print(
                f"{mode:8} | {domain:17} | {query:45} | "
                f"recall@{args.top_k}={metrics['recall_at_k']:.3f} "
                f"mrr@{args.top_k}={metrics['mrr_at_k']:.3f} "
                f"latency_ms={latency_ms:.2f} trace_id={trace_id}"
            )

    deadline = time.monotonic() + 60
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_ids[0]) if trace_ids else None
    if time.monotonic() > deadline:
        raise TimeoutError("MLflow trace readback exceeded the 60-second verification deadline")
    if trace is None or not trace.data.spans:
        raise RuntimeError(f"MLflow trace was not persisted: {trace_ids[0]}")
    span_names = [span.name for span in trace.data.spans]
    required_spans = {"evaluate_retrieval_query", "lexical_retrieval", "compute_relevance_metrics"}
    if not required_spans.issubset(span_names):
        raise RuntimeError(f"Persisted trace is missing expected spans: {span_names}")

    print(
        f"Verified persisted trace: id={trace.info.trace_id} "
        f"experiment_id={trace.info.experiment_id} spans={span_names}"
    )
    print(f"Logged {len(trace_ids)} traces to experiment {experiment.name!r}.")


if __name__ == "__main__":
    main()
