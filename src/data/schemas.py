from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import REQUIRED_FILES, TRANSACTION_REQUIRED_COLUMNS

REQUIRED_FILE_NAMES = set(REQUIRED_FILES.values())
TRANSACTION_COLUMNS = TRANSACTION_REQUIRED_COLUMNS


def validate_transactions_schema(df: pd.DataFrame) -> None:
    missing = [c for c in TRANSACTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"transactions.csv missing required columns: {missing}")
    if df.empty:
        raise ValueError("transactions.csv is empty")


def validate_json_list(name: str, rows: list[dict], required_keys: set[str]) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a JSON list")
    if not rows:
        raise ValueError(f"{name} is empty")
    for idx, row in enumerate(rows[:20]):
        if not isinstance(row, dict):
            raise ValueError(f"{name}[{idx}] is not a JSON object")
        missing = sorted(required_keys - set(row.keys()))
        if missing:
            raise ValueError(f"{name}[{idx}] missing keys: {missing}")


def assert_required_files(dataset_root: str) -> None:
    root = Path(dataset_root)
    missing = [f for f in REQUIRED_FILE_NAMES if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset root '{dataset_root}' is missing required files: {missing}"
        )
