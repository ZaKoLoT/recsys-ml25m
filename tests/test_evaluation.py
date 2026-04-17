import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.recsys import evaluation


def test_evaluate_model_returns_metrics_for_k_10_and_k_20():
    split_df = pd.DataFrame({"user_id": [1, 2], "item_id": [3, 4]})
    train_seen = {1: {1, 2}, 2: {2, 3}}
    recommend_fn = lambda _, seen, k: [item for item in [3, 4, 1, 2] if item not in seen][:k]  # noqa: E731

    results_10 = evaluation.evaluate_model(split_df, train_seen, recommend_fn, k=10)
    results_20 = evaluation.evaluate_model(split_df, train_seen, recommend_fn, k=20)

    assert results_10["n_users"] == 2
    assert results_10["Recall@10"] == 1.0
    assert results_10["nDCG@10"] == 1.0

    assert results_20["n_users"] == 2
    assert results_20["Recall@20"] == 1.0
    assert results_20["nDCG@20"] == 1.0


def test_write_report_writes_all_requested_metrics(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(evaluation, "REPORT_DIR", report_dir)

    val_results = {
        "n_users": 2,
        "Recall@10": 0.5,
        "nDCG@10": 0.3,
        "Recall@20": 0.75,
        "nDCG@20": 0.4,
    }
    test_results = {
        "n_users": 2,
        "Recall@10": 0.6,
        "nDCG@10": 0.35,
        "Recall@20": 0.8,
        "nDCG@20": 0.45,
    }

    evaluation.write_report(
        model_name="baseline_popularity",
        val_results=val_results,
        test_results=test_results,
        ks=[10, 20],
        params={
            "K": "10, 20",
            "method": "global item interaction count on train",
            "leakage": "train items excluded from recommendations",
        },
    )

    report_path = report_dir / "baseline_popularity.md"
    content = report_path.read_text(encoding="utf-8")

    assert "| Recall@10 | 0.5 | 0.6 |" in content
    assert "| nDCG@10 | 0.3 | 0.35 |" in content
    assert "| Recall@20 | 0.75 | 0.8 |" in content
    assert "| nDCG@20 | 0.4 | 0.45 |" in content
