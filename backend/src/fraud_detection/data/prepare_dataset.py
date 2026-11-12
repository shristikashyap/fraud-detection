"""CLI to download and prepare the public OpenML fraud dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fraud_detection.data.ingestion import download_openml_creditcard, load_transactions
from fraud_detection.data.preprocessing import prepare_splits
from fraud_detection.data.source import (
    CITATION,
    DATASET_NAME,
    DATASET_PAGE,
    EXPECTED_ROWS,
    FRAUD_ROWS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DATA_PATH = PROJECT_ROOT / "backend" / "data" / "raw" / "openml_creditcard.parquet"
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"


def prepare_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_dir: Path = PROCESSED_DIR,
    *,
    download: bool = False,
) -> dict[str, object]:
    if download:
        raw_path = download_openml_creditcard(raw_path)

    raw = load_transactions(raw_path)
    splits = prepare_splits(raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    for name, frame in split_frames.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    report: dict[str, object] = {
        "dataset": DATASET_NAME,
        "source": DATASET_PAGE,
        "citation": CITATION,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(raw),
        "source_expected_rows": EXPECTED_ROWS,
        "source_expected_fraud_rows": FRAUD_ROWS,
        "split_method": "chronological by source Time; stable ordering for ties",
        "label": "Class (1 = fraud, 0 = non-fraud)",
        "feature_columns": list(splits.train.columns.drop("is_fraud")),
        "splits": {
            name: {
                "rows": len(frame),
                "fraud_rows": int(frame["is_fraud"].sum()),
                "non_fraud_rows": int((frame["is_fraud"] == 0).sum()),
                "path": str(output_dir / f"{name}.parquet"),
            }
            for name, frame in split_frames.items()
        },
        "train_only_median_imputation": splits.imputation_values,
        "model_metrics": None,
    }
    report_path = output_dir / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the public OpenML dataset if it is not already present.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_DATA_PATH,
        help="Existing local CSV or Parquet input path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for prepared Parquet splits and the JSON report.",
    )
    arguments = parser.parse_args()
    report = prepare_dataset(
        arguments.raw_path, arguments.output_dir, download=arguments.download
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
