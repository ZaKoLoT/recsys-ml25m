import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")


def calculate_sparsity(df: pd.DataFrame, user_col: str, item_col: str) -> float:
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    return 1.0 - (len(df) / (n_users * n_items)) if n_users and n_items else 1.0


def generate_v0_split(
    df: pd.DataFrame,
    n_test: int,
    n_val: int,
    n_train_min: int,
    user_col: str,
    item_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_required = n_test + n_val + n_train_min

    df = df[df.groupby(user_col)[user_col].transform("count") >= min_required].copy()

    df = df.sort_values(by=[user_col, time_col]).copy()

    df["reverse_rank"] = df.groupby(user_col).cumcount(ascending=False)

    test_mask = df["reverse_rank"] < n_test

    test_df = df[test_mask].drop(columns=["reverse_rank"])
    train_df = df[~test_mask].drop(columns=["reverse_rank"])

    return train_df, test_df


def run_checks(train_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str, time_col: str):
    assert len(set(train_df.index).intersection(set(test_df.index))) == 0, "Overlap detected"

    train_users = set(train_df[user_col].unique())
    test_users = set(test_df[user_col].unique())
    assert test_users.issubset(train_users), "Test users missing from train set"

    max_train_ts = train_df.groupby(user_col)[time_col].max()
    min_test_ts = test_df.groupby(user_col)[time_col].min()

    # Check temporal logic only for users present in test
    test_users_list = list(test_users)
    assert (max_train_ts.loc[test_users_list] <= min_test_ts.loc[test_users_list]).all(), (
        "Temporal leakage detected"
    )


def print_split_stats(train_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str, item_col: str):
    stats = f"""
### Split V0 Statistics

| Metric | Train | Test |
| :--- | :--- | :--- |
| Users | {train_df[user_col].nunique():,} | {test_df[user_col].nunique():,} |
| Items | {train_df[item_col].nunique():,} | {test_df[item_col].nunique():,} |
| Interactions | {len(train_df):,} | {len(test_df):,} |
| Sparsity | {calculate_sparsity(train_df, user_col, item_col):.4%} | {calculate_sparsity(test_df, user_col, item_col):.4%} |
    """
    logging.info(stats)


def main():
    INPUT_FILE = Path("data/processed/interactions.parquet")
    TRAIN_OUTPUT = Path("data/processed/interactions_train_v0.parquet")
    TEST_OUTPUT = Path("data/processed/interactions_test_v0.parquet")

    USER_COL = "user_id"
    ITEM_COL = "movie_id"
    TIME_COL = "timestamp"
    N_TEST = 5
    N_VAL = 2
    N_TRAIN_MIN = 3

    logging.info("Loading MovieLens data...")
    df = pd.read_parquet(INPUT_FILE)

    logging.info("Generating split...")
    train_df, test_df = generate_v0_split(
        df, N_TEST, N_VAL, N_TRAIN_MIN, USER_COL, ITEM_COL, TIME_COL
    )

    logging.info("Running integrity checks...")
    run_checks(train_df, test_df, USER_COL, TIME_COL)
    logging.info("All checks passed.")

    logging.info("Exporting to Parquet format...")
    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_OUTPUT, index=False)
    test_df.to_parquet(TEST_OUTPUT, index=False)

    print_split_stats(train_df, test_df, USER_COL, ITEM_COL)


if __name__ == "__main__":
    main()
