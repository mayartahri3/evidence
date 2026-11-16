"""CLI: python -m pipelines.ingestion.run --max-per-query 150"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from backend.app.config import ROOT, load_env, load_project_config
from pipelines.ingestion.dedup import deduplicate
from pipelines.ingestion.openalex import OpenAlexClient, work_to_paper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-query", type=int, default=150)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    cfg = load_project_config()
    src = cfg["sources"]["openalex"]
    version = cfg["project"]["processing_version"]
    out = Path(args.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "processed").mkdir(parents=True, exist_ok=True)

    env = load_env()
    if not env.openalex_api_key:
        print("WARNING: no OPENALEX_API_KEY set; keyless usage has a very small daily budget.")
    client = OpenAlexClient(api_key=env.openalex_api_key, base_url=src["base_url"])

    papers, per_query = [], []
    for domain, queries in cfg["scope"]["queries"].items():
        for q in queries:
            raw = list(client.search_works(
                q, cfg["scope"]["year_from"], src["work_types"], args.max_per_query, src["per_page"]))
            # keep the untouched API response so the corpus can be rebuilt without re-querying
            slug = f"{domain}__{q}".replace(" ", "_")
            (out / "raw" / f"openalex_{slug}.jsonl").write_text(
                "\n".join(json.dumps(w) for w in raw), encoding="utf-8")
            converted = [p for w in raw if (p := work_to_paper(w, domain, version))]
            papers += converted
            per_query.append({"domain": domain, "query": q, "fetched": len(raw), "usable": len(converted)})
            print(f"{domain:17} | {q:45} | fetched={len(raw)} usable={len(converted)}")

    unique, merge_log, possible = deduplicate(papers)
    (out / "processed" / "papers.jsonl").write_text(
        "\n".join(p.model_dump_json() for p in unique), encoding="utf-8")
    (out / "processed" / "possible_duplicates.json").write_text(
        json.dumps(possible, indent=2), encoding="utf-8")
    manifest = {  # provenance sheet (spec 6.4 step 6)
        "source": "OpenAlex", "collected_on": date.today().isoformat(),
        "processing_version": version, "queries": per_query,
        "total_before_dedup": len(papers), "total_after_dedup": len(unique),
        "merged": len(merge_log), "possible_duplicates_flagged": len(possible),
        "with_abstract": sum(1 for p in unique if p.abstract),
        "with_doi": sum(1 for p in unique if p.doi),
        "with_open_access_link": sum(1 for p in unique if p.full_text_status == "open_access_link"),
    }
    (out / "processed" / "ingestion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "queries"}, indent=2))


if __name__ == "__main__":
    main()
