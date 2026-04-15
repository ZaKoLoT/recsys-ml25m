import datetime
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def evaluate_model(
    split_df: pd.DataFrame,
    train_seen: dict[int, set],
    recommend_fn: Callable[[int, set, int], list],
    k: int,
) -> dict:
    """Evaluates a recommendation model on a split.

    For each user in the split, calls recommend_fn to get top-K items and
    computes Recall@K and nDCG@K against the ground truth.

    Args:
        split_df: Val or test interactions with columns user_id, item_id.
        train_seen: Dict mapping user_id -> set of item_ids seen in train.
        recommend_fn: Callable(user_id, seen_items, k) -> list of item_ids.
        k: Cutoff for metrics and recommendations.

    Returns:
        dict: n_users, Recall@K, nDCG@K (means across users).
    """
    ground_truth = split_df.groupby("user_id")["item_id"].apply(set)
    users = ground_truth.index.tolist()

    # Precompute log2 denominators once for all users
    log2_ranks = np.log2(np.arange(2, k + 2))
    idcg_perfect = (1.0 / log2_ranks).cumsum()  # idcg_perfect[i] = IDCG for i+1 relevant items

    recalls = np.zeros(len(users))
    ndcgs = np.zeros(len(users))

    for i, user_id in enumerate(users):
        seen = train_seen.get(user_id, set())
        recs = recommend_fn(user_id, seen, k)
        relevant = ground_truth[user_id]
        if not relevant:
            continue

        hits = np.array([1.0 if item in relevant else 0.0 for item in recs[:k]])
        recalls[i] = hits.sum() / min(len(relevant), k)
        dcg = (hits / log2_ranks[: len(hits)]).sum()
        idcg = idcg_perfect[min(len(relevant), k) - 1]
        ndcgs[i] = dcg / idcg

    return {
        "n_users": len(users),
        f"Recall@{k}": round(float(recalls.mean()), 4),
        f"nDCG@{k}": round(float(ndcgs.mean()), 4),
    }


def write_report(
    model_name: str,
    val_results: dict,
    test_results: dict,
    k: int,
    params: dict,
):
    """Writes evaluation results to reports/<model_name>.md.

    Args:
        model_name: Name used for the report filename (e.g. "baseline_popularity").
        val_results: Metrics dict for the validation split.
        test_results: Metrics dict for the test split.
        k: Cutoff used for evaluation.
        params: Dict of model/eval parameters to document in the report.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{model_name}.md"

    with open(path, "w") as f:
        f.write(f"# {model_name.replace('_', ' ').title()}\n\n")
        f.write(f"*Generated: {datetime.datetime.now().isoformat(timespec='seconds')}*\n\n")
        f.write("## Parameters\n\n")
        for key, value in params.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n## Results\n\n")
        f.write(
            f"| Metric | Val ({val_results['n_users']:,} users)"
            f" | Test ({test_results['n_users']:,} users) |\n"
        )
        f.write("| :--- | :--- | :--- |\n")
        for key in [f"Recall@{k}", f"nDCG@{k}"]:
            f.write(f"| {key} | {val_results[key]} | {test_results[key]} |\n")

    logging.info(f"Report written to {path}")
