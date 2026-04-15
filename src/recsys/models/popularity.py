import pandas as pd


def fit_popularity(train_df: pd.DataFrame) -> pd.Series:
    """Ranks items by number of interactions in the training set.

    Args:
        train_df: Training interactions with columns user_id, item_id.

    Returns:
        pd.Series: Item counts sorted descending, indexed by item_id.
    """
    return train_df["item_id"].value_counts()


def recommend(
    ranking: pd.Series,
    seen_items: set,
    k: int,
) -> list:
    """Returns the top-k most popular items not seen by the user.

    Only scans as many items as needed (k + seen_items) rather than the full
    ranking, which avoids O(n_items) iteration per user.

    Args:
        ranking: Output of fit_popularity — item counts sorted descending.
        seen_items: Set of item_ids already seen by the user (from train).
        k: Number of items to recommend.

    Returns:
        list: Up to k item_ids, ordered by popularity.
    """
    buffer = min(len(ranking), k + len(seen_items) + 100)
    return [item for item in ranking.index[:buffer] if item not in seen_items][:k]
