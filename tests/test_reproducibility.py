"""Reproducibility check for make_dataset.py.

Runs the dataset build twice and verifies that every produced Parquet file has
identical content across runs. Row order is ignored (rows are canonically
sorted before hashing), so this catches real data differences without being
tripped by non-deterministic row ordering from DuckDB DISTINCT, pandas
unstable sort, or Python hash randomization.

The metadata.json is compared separately, excluding the `created_at` field
which is expected to differ.

This test is slow (it rebuilds the full dataset twice). Run it explicitly with:
    pytest tests/test_reproducibility.py -s
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "data" / "reports"
CONFIG_PATH = PROJECT_DIR / "configs" / "dataset_V1.yaml"
SCRIPT_PATH = PROJECT_DIR / "scripts" / "make_dataset.py"


def _content_hash(path: Path) -> str:
    """Hash Parquet content ignoring row order.

    Loads the file, sorts rows by all columns (stable), then hashes the
    row-wise pandas hash. Two files with the same rows in different order
    produce the same hash.
    """
    df = pd.read_parquet(path)
    df = df.sort_values(by=list(df.columns), kind="stable", ignore_index=True)
    row_hashes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    h = hashlib.sha256()
    h.update(",".join(df.columns).encode())
    h.update(b"|")
    h.update(row_hashes)
    return h.hexdigest()


def _hash_parquets(directory: Path) -> dict[str, str]:
    return {p.name: _content_hash(p) for p in sorted(directory.glob("*.parquet"))}


def _load_metadata_without_timestamp() -> dict:
    with open(REPORTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    meta.pop("created_at", None)
    return meta


def _run_make_dataset():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(CONFIG_PATH)],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"make_dataset.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.slow
def test_make_dataset_is_reproducible():
    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
    assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"

    _run_make_dataset()
    hashes_run1 = _hash_parquets(PROCESSED_DIR)
    metadata_run1 = _load_metadata_without_timestamp()

    _run_make_dataset()
    hashes_run2 = _hash_parquets(PROCESSED_DIR)
    metadata_run2 = _load_metadata_without_timestamp()

    assert set(hashes_run1) == set(hashes_run2), (
        f"File set differs between runs:\n"
        f"  only in run1: {set(hashes_run1) - set(hashes_run2)}\n"
        f"  only in run2: {set(hashes_run2) - set(hashes_run1)}"
    )

    mismatches = {
        name: (hashes_run1[name], hashes_run2[name])
        for name in hashes_run1
        if hashes_run1[name] != hashes_run2[name]
    }
    assert not mismatches, (
        "Parquet content differs between runs (non-reproducible, row order ignored):\n"
        + "\n".join(f"  {name}: {h1} != {h2}" for name, (h1, h2) in mismatches.items())
    )

    assert metadata_run1 == metadata_run2, (
        "metadata.json differs between runs (excluding created_at)"
    )
