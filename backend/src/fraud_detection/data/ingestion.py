"""Download and load labeled transaction data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from fraud_detection.data.source import DATASET_URL

_PARQUET_MAGIC = b"PAR1"


def download_openml_creditcard(destination: Path, *, timeout: int = 60) -> Path:
    """Download the public OpenML Parquet file without replacing a valid file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and _is_parquet_file(destination):
        return destination

    request = Request(DATASET_URL, headers={"User-Agent": "fraud-detection-capstone/0.1"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"OpenML dataset download failed with HTTP {response.status}."
                )
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

        if not _is_parquet_file(temporary_path):
            raise RuntimeError("Downloaded OpenML file is not a valid Parquet file.")

        os.replace(temporary_path, destination)
        return destination
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Unable to download the OpenML dataset: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_transactions(path: Path) -> pd.DataFrame:
    """Load a labeled CSV or Parquet transaction dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Transaction dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Use CSV or Parquet."
        )

    if frame.empty:
        raise ValueError(f"Transaction dataset is empty: {path}")
    return frame


def _is_parquet_file(path: Path) -> bool:
    try:
        if path.stat().st_size < 8:
            return False
        with path.open("rb") as file:
            start = file.read(4)
            file.seek(-4, os.SEEK_END)
            end = file.read(4)
        return start == _PARQUET_MAGIC and end == _PARQUET_MAGIC
    except OSError:
        return False
