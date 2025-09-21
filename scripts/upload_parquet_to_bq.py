"""Upload GOLD parquet tables to BigQuery (dataset retail_gold)."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from google.cloud import bigquery  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "google-cloud-bigquery debe estar instalado para ejecutar este script."
    ) from exc

from utils.io import get_paths, logger

PATHS = get_paths()
DEFAULT_DATASET = "retail_gold"


def upload(dataset: str = DEFAULT_DATASET, project: str | None = None, location: str | None = None) -> None:
    client = bigquery.Client(project=project)
    gold_dir = PATHS.gold
    if not gold_dir.exists():
        raise FileNotFoundError(f"No existe el directorio GOLD: {gold_dir}")

    for parquet in sorted(gold_dir.glob("*.parquet")):
        table_id = f"{dataset}.{parquet.stem}"
        logger.info("Subiendo %s → %s", parquet.name, table_id)
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        with parquet.open("rb") as handle:
            load_job = client.load_table_from_file(
                handle,
                table_id,
                location=location,
                job_config=job_config,
            )
        load_job.result()
        logger.info("%s filas cargadas en %s", load_job.output_rows, table_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sube Parquet GOLD a BigQuery")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset destino (default retail_gold)")
    parser.add_argument("--project", default=None, help="ID de proyecto GCP")
    parser.add_argument("--location", default=None, help="Región BigQuery")
    args = parser.parse_args(argv)

    upload(dataset=args.dataset, project=args.project, location=args.location)


if __name__ == "__main__":
    main()
