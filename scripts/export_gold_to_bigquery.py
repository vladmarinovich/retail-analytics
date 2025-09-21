"""Export GOLD parquet tables to BigQuery."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from utils.io import get_paths, logger  # noqa: E402


def _load_bigquery_client(project: str | None = None):
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "google-cloud-bigquery is required for this script."
        ) from exc
    return bigquery


def export(dataset: str, project: str | None = None, location: str | None = None) -> None:
    bigquery = _load_bigquery_client()
    client = bigquery.Client(project=project)
    gold_dir = get_paths().gold
    for parquet in sorted(gold_dir.glob("*.parquet")):
        table_id = f"{dataset}.{parquet.stem}"
        logger.info("Uploading %s to %s", parquet.name, table_id)
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        with parquet.open("rb") as handle:
            load_job = client.load_table_from_file(
                handle,
                table_id=table_id,
                location=location,
                job_config=job_config,
            )
        load_job.result()
        logger.info("Loaded %s rows into %s", load_job.output_rows, table_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gold tables to BigQuery")
    parser.add_argument("dataset", help="BigQuery dataset name", default="retail_gold")
    parser.add_argument("--project", help="GCP project ID", default=None)
    parser.add_argument("--location", help="BigQuery location", default=None)
    args = parser.parse_args()
    export(args.dataset, project=args.project, location=args.location)


if __name__ == "__main__":
    main()
