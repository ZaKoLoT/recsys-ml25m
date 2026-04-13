from pathlib import Path

import pandas as pd

DATA_PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the V1 train, val and test interaction splits.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test dataframes.
            Each has columns: user_id, movie_id, timestamp, interaction.
    """
    train = pd.read_parquet(DATA_PROCESSED_DIR / "interactions_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "interactions_val.parquet")
    test = pd.read_parquet(DATA_PROCESSED_DIR / "interactions_test.parquet")
    return train, val, test


def load_items() -> pd.DataFrame:
    """Loads the items table.

    Returns:
        pd.DataFrame: Items with columns: item_id, title, year, genres.
    """
    return pd.read_parquet(DATA_PROCESSED_DIR / "items.parquet")


def load_tags() -> pd.DataFrame:
    """Loads the item tags table.

    Returns:
        pd.DataFrame: Item tags with columns: movie_id, tags.
    """
    return pd.read_parquet(DATA_PROCESSED_DIR / "item_tags.parquet")


def load_text() -> pd.DataFrame:
    """Loads the item text table (title + genres + tags concatenated).

    Returns:
        pd.DataFrame: Item text with columns: movie_id, text.
    """
    return pd.read_parquet(DATA_PROCESSED_DIR / "item_text.parquet")
