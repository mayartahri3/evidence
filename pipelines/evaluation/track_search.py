from __future__ import annotations

import argparse
import time

import mlflow

from backend.app.config import ROOT, load_project_config
from pipelines.indexing import CorpusIndex, SemanticIndex, load_processed_papers
from pipelines.indexing.search import PROCESSED_PAPERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Log search baseline runs to MLflow.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    papers = load_processed_papers()
    if not papers:
        raise RuntimeError(f"No processed papers found at {PROCESSED_PAPERS}")

    cfg = load_project_config()
    lexical_index = CorpusIndex(papers)
    semantic_index = SemanticIndex(papers, cfg["models"]["embedding"])
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("evidencemap-retrieval-comparison")

    for domain, queries in cfg["scope"]["queries"].items():
        for query in queries:
            for mode, index in (
                ("lexical", lexical_index),
                ("semantic", semantic_index),
            ):
                started = time.perf_counter()
                results = index.search(query, limit=args.top_k, domain=domain)
                latency_ms = (time.perf_counter() - started) * 1000

                with mlflow.start_run(run_name=f"{mode}: {domain}: {query}"):
                    mlflow.log_params({
                        "mode": mode,
                        "domain": domain,
                        "query": query,
                        "top_k": args.top_k,
                        "corpus_size": len(papers),
                        "processing_version": cfg["project"]["processing_version"],
                    })
                    mlflow.log_metrics({
                        "result_count": len(results),
                        "search_latency_ms": latency_ms,
                    })
                    if mode == "semantic":
                        mlflow.log_param("embedding_model", semantic_index.model_name)
                        mlflow.log_metric(
                            "embedding_index_build_seconds", semantic_index.build_seconds
                        )
                    mlflow.log_dict({"results": results}, "top_results.json")

                print(
                    f"{mode:8} | {domain:17} | {query:45} | "
                    f"results={len(results)} latency_ms={latency_ms:.2f}"
                )


if __name__ == "__main__":
    main()
