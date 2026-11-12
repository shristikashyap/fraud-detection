import pandas as pd
import pytest

from fraud_detection.data.ingestion import load_transactions


def test_load_transactions_reads_csv(tmp_path):
    source = tmp_path / "transactions.csv"
    pd.DataFrame({"Time": [0], "Amount": [1.0], "Class": [0]}).to_csv(source, index=False)

    result = load_transactions(source)

    assert result.to_dict(orient="records") == [
        {"Time": 0, "Amount": 1.0, "Class": 0}
    ]


def test_load_transactions_rejects_unsupported_format(tmp_path):
    source = tmp_path / "transactions.txt"
    source.write_text("not a supported dataset", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV or Parquet"):
        load_transactions(source)
