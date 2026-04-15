import math


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Computes Recall@K.

    Args:
        recommended: Ordered list of recommended item_ids (length <= k).
        relevant: Set of ground-truth item_ids.
        k: Cutoff.

    Returns:
        float: Recall@K in [0, 1].
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Computes nDCG@K.

    Args:
        recommended: Ordered list of recommended item_ids (length <= k).
        relevant: Set of ground-truth item_ids.
        k: Cutoff.

    Returns:
        float: nDCG@K in [0, 1].
    """
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant
    )
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0
