import argparse
import datetime
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Loads parameters from a YAML configuration file and converts them into a dictionary.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration parameters.
    """
    with open(config_path) as file:
        return yaml.safe_load(file)


logging.basicConfig(level=logging.INFO, format="%(message)s")


# Dictionary containing SQL queries for specific file conversions
SPECIFIC_CONVERSIONS = {
    # Conversion for movies.csv to extract year and clean title
    "movies.csv": r"""
        SELECT
            DISTINCT
            CAST(movieId AS INTEGER) AS item_id,
            CAST(TRIM(regexp_replace(title, '\s*\(\d{{4}}\)\s*$', '')) AS VARCHAR) AS title,
            TRY_CAST(regexp_extract(title, '\((\d{{4}})\)\s*$', 1) AS INTEGER) AS year,
            CAST(genres as VARCHAR) AS genres
        FROM read_csv_auto('{input}')
    """,
    # Conversion for links.csv to ensure correct data types
    "links.csv": r"""
        SELECT CAST(movieId AS INTEGER) AS movie_id, imdbId, tmdbId
        FROM read_csv_auto('{input}')
        WHERE tmdbId IS NOT NULL
    """,
    # Conversion for tags.csv to ensure correct data types
    "tags.csv": r"""
        SELECT
            CAST(userId AS INTEGER) AS user_id,
            CAST(movieId AS INTEGER) AS item_id,
            CAST(LOWER(TRIM(tag)) AS VARCHAR) AS tag,
            CAST(timestamp AS INTEGER) AS timestamp
        FROM read_csv_auto('{input}')
    """,
    # Conversion for genome-tags.csv to ensure correct data types
    "genome-tags.csv": r"""
        SELECT
            CAST(tagId AS INTEGER) AS tag_id,
            CAST(tag AS VARCHAR) AS tag
        FROM read_csv_auto('{input}')
    """,
    # Conversion for genome-scores.csv to ensure correct data types
    "genome-scores.csv": r"""
        SELECT
            CAST(movieId AS INTEGER) AS item_id,
            CAST(tagId AS INTEGER) AS tag_id,
            CAST(relevance AS FLOAT) AS relevance
        FROM read_csv_auto('{input}')
    """,
}

# List of CSV files that have been specifically processed
TRAITED_FILES = [
    "ratings.csv",
    "movies.csv",
    "links.csv",
    "tags.csv",
    "genome-tags.csv",
    "genome-scores.csv",
]


def parse_arguments():
    """Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing the config path.
    """
    parser = argparse.ArgumentParser(description="Process raw MovieLens data.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    return parser.parse_args()


def load_and_extract_config(config_path: str):
    """Loads configuration and extracts parameters.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        tuple: (config dict, rating_threshold, min_user_inter, min_item_inter, n_test, n_val, n_train)
    """
    config = load_config(config_path)
    rating_threshold = config.get("rating_threshold", 4.0)
    min_user_inter = config.get("min_user_interactions", 5)
    min_item_inter = config.get("min_item_interactions", 5)
    n_test = config.get("n_test", 5)
    n_val = config.get("n_val", 2)
    n_train = config.get("n_train", 3)
    return config, rating_threshold, min_user_inter, min_item_inter, n_test, n_val, n_train


def log_parameters(rating_threshold, min_user_inter, min_item_inter, n_test, n_val, n_train):
    """Logs the configuration parameters.

    Args:
        rating_threshold (float): Minimum rating threshold.
        min_user_inter (int): Minimum user interactions.
        min_item_inter (int): Minimum item interactions.
        n_test (int): Number of test interactions.
        n_val (int): Number of validation interactions.
        n_train (int): Number of train interactions.
    """
    logging.info("--- Configuration Parameters ---")
    logging.info(f"Rating threshold: {rating_threshold}")
    logging.info(f"Min user interactions: {min_user_inter}")
    logging.info(f"Min item interactions: {min_item_inter}")
    logging.info(f"N Test: {n_test} | N Val: {n_val} | N Train: {n_train}")
    logging.info("--------------------------------")


def setup_directories():
    """Sets up project directories and database connection.

    Returns:
        tuple: (raw_dir, processed_dir, con)
    """
    project_dir = Path(__file__).resolve().parent.parent
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    return raw_dir, processed_dir, con


def filter_interactions_iteratively(
    df: pd.DataFrame, min_user_inter: int, min_item_inter: int
) -> pd.DataFrame:
    """
    Iteratively filters users and items with fewer than the specified minimum interactions.
    The loop continues until the size of the dataframe remains stable (convergence).

    Args:
        df (pd.DataFrame): The input dataframe containing interactions.
        min_user_inter (int): Minimum number of interactions required for a user to be retained.
        min_item_inter (int): Minimum number of interactions required for an item to be retained.
    Returns:
        pd.DataFrame: The filtered dataframe after convergence.
    """
    logging.info("Starting iterative filtering (K-core reduction)...")

    iteration = 1
    while True:
        start_len = len(df)

        # 1. Filter out rare items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_inter].index
        df = df[df["item_id"].isin(valid_items)]

        # 2. Filter out rare users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_inter].index
        df = df[df["user_id"].isin(valid_users)]

        end_len = len(df)
        logging.info(f"Iteration {iteration}: dropped {start_len - end_len} rows.")

        # 3. Check for convergence (if no rows were dropped, we stop)
        if start_len == end_len:
            logging.info("Filtering converged!")
            break

        iteration += 1

    return df


def print_filtering_report(df_before: pd.DataFrame, df_after: pd.DataFrame):
    """
    Prints a report comparing the dataset before and after filtering.

    Args:
        df_before (pd.DataFrame): The dataframe before filtering.
        df_after (pd.DataFrame): The dataframe after filtering.

    Returns:
        None
    """
    users_b, items_b, inter_b = (
        df_before["user_id"].nunique(),
        df_before["item_id"].nunique(),
        len(df_before),
    )
    users_a, items_a, inter_a = (
        df_after["user_id"].nunique(),
        df_after["item_id"].nunique(),
        len(df_after),
    )

    pct_dropped = ((inter_b - inter_a) / inter_b) * 100

    # Helper function to compute min, median, and max interactions per user/item
    def get_stats(df, column):
        counts = df[column].value_counts()
        return counts.min(), counts.median(), counts.max()

    u_min_b, u_med_b, u_max_b = get_stats(df_before, "user_id")
    u_min_a, u_med_a, u_max_a = get_stats(df_after, "user_id")

    i_min_b, i_med_b, i_max_b = get_stats(df_before, "item_id")
    i_min_a, i_med_a, i_max_a = get_stats(df_after, "item_id")

    logging.info("=== Data Filtering Report (Before -> After) ===")
    logging.info(f"Users: {users_b} -> {users_a}")
    logging.info(f"Items: {items_b} -> {items_a}")
    logging.info(f"Interactions: {inter_b} -> {inter_a}")
    logging.info(f"Total interactions removed: {pct_dropped:.2f}%")

    logging.info("--- User Interactions Stats ---")
    logging.info(f"Before: Min {u_min_b} | Median {u_med_b:.1f} | Max {u_max_b}")
    logging.info(f"After : Min {u_min_a} | Median {u_med_a:.1f} | Max {u_max_a}")

    logging.info("--- Item Interactions Stats ---")
    logging.info(f"Before: Min {i_min_b} | Median {i_med_b:.1f} | Max {i_max_b}")
    logging.info(f"After : Min {i_min_a} | Median {i_med_a:.1f} | Max {i_max_a}")
    logging.info("===============================================")


def collect_filtering_stats(df_before, df_after) -> dict:
    """
    Collects filtering statistics.

    Args:
        df_before (pd.DataFrame): Dataframe before filtering.
        df_after (pd.DataFrame): Dataframe after filtering.

    Returns:
        dict: Filtering stats (users, items, interactions before/after).
    """
    return {
        "users": {
            "before": int(df_before["user_id"].nunique()),
            "after": int(df_after["user_id"].nunique()),
        },
        "items": {
            "before": int(df_before["item_id"].nunique()),
            "after": int(df_after["item_id"].nunique()),
        },
        "interactions": {"before": len(df_before), "after": len(df_after)},
        "pct_removed": round(((len(df_before) - len(df_after)) / len(df_before)) * 100, 2),
    }


def process_ratings(con, raw_dir, processed_dir, rating_threshold, min_user_inter, min_item_inter):
    """Processes ratings.csv to create interactions.parquet.

    Args:
        con: DuckDB connection.
        raw_dir (Path): Path to raw data directory.
        processed_dir (Path): Path to processed data directory.
        rating_threshold (float): Minimum rating to consider as interaction.
        min_user_inter (int): Minimum user interactions for K-core filtering.
        min_item_inter (int): Minimum item interactions for K-core filtering.
    """
    ratings_csv = raw_dir / "ratings.csv"
    interactions_pq = processed_dir / "interactions.parquet"

    if ratings_csv.exists():
        # Validation and conversion of ratings.csv to interactions.parquet
        count_expected = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{ratings_csv}') WHERE rating >= {rating_threshold}"
        ).fetchone()[0]
        query_interactions = f"""
        COPY (
            SELECT
                CAST(userId AS INTEGER) AS user_id,
                CAST(movieId AS INTEGER) AS item_id,
                CAST(timestamp AS INTEGER) AS timestamp,
                1 AS interaction
            FROM read_csv_auto('{ratings_csv}')
            WHERE rating >= {rating_threshold}
            ORDER BY timestamp
        ) TO '{interactions_pq}' (FORMAT PARQUET)
        """
        con.execute(query_interactions)

        count_written = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{interactions_pq}')"
        ).fetchone()[0]
        assert count_written == count_expected, (
            f"Line error : Expected {count_expected}, obtained {count_written}"
        )
        print(f"      - Row count check passed ({count_written} rows).")

        min_ts = con.execute(
            f"SELECT MIN(timestamp) FROM read_parquet('{interactions_pq}')"
        ).fetchone()[0]
        assert min_ts >= 0, f"Error : Negative timestamp detected ({min_ts})"
        print("      - Timestamp check passed (no negative values).")

        distinct_interactions = con.execute(
            f"SELECT DISTINCT interaction FROM read_parquet('{interactions_pq}')"
        ).fetchall()
        unique_values = {row[0] for row in distinct_interactions}
        assert unique_values.issubset({0, 1}), (
            f"Error : Unauthorized interaction values {unique_values}"
        )
        print(f"      - Interaction values check passed (found: {unique_values}).")

        print("All assertions passed successfully")

        logging.info("Loading data into Pandas for iterative filtering...")
        df_interactions = pd.read_parquet(interactions_pq)
        df_raw_interactions = df_interactions.copy()
        df_filtered_interactions = filter_interactions_iteratively(
            df=df_interactions, min_user_inter=min_user_inter, min_item_inter=min_item_inter
        )
        print_filtering_report(df_before=df_raw_interactions, df_after=df_filtered_interactions)
        logging.info("Saving filtered data back to Parquet...")
        df_filtered_interactions.to_parquet(interactions_pq, index=False)

        filtering_stats = collect_filtering_stats(
            df_before=df_raw_interactions,
            df_after=df_filtered_interactions,
        )
        return filtering_stats
    else:
        print(f"File {ratings_csv} not found")
        return {}


def process_specific_files(con, raw_dir, processed_dir, specific_conversions):
    """Processes specific CSV files with custom SQL transformations.

    Args:
        con: DuckDB connection.
        raw_dir (Path): Path to raw data directory.
        processed_dir (Path): Path to processed data directory.
        specific_conversions (dict): Dictionary of filename to SQL query.
    """
    for filename, sql_select in specific_conversions.items():
        input_csv = raw_dir / filename
        if input_csv.exists():
            print(f"Converting {filename}...")
            if filename == "movies.csv":
                output_name = "items.parquet"
            else:
                output_name = filename.replace(".csv", ".parquet")
            output_pq = processed_dir / output_name
            query = f"COPY ({sql_select.format(input=input_csv)}) TO '{output_pq}' (FORMAT PARQUET)"
            con.execute(query)


def generate_v1_split(
    df: pd.DataFrame,
    n_test: int,
    n_val: int,
    n_train: int,
    user_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Generates a temporal train/val/test split per user.

    Each user's interactions are sorted by time. The last n_test interactions
    go to test, the n_val before that go to val, and the rest go to train.
    Users with fewer than n_test + n_val + n_train interactions are excluded and logged.

    Args:
        df (pd.DataFrame): Filtered interactions dataframe.
        n_test (int): Number of interactions per user reserved for test.
        n_val (int): Number of interactions per user reserved for validation.
        n_train (int): Minimum number of interactions per user required for train.
        user_col (str): Name of the user column.
        time_col (str): Name of the timestamp column.

    Returns:
        tuple: (train_df, val_df, test_df, n_excluded) where n_excluded is the
               number of users dropped for having too few interactions.
    """
    min_required = n_test + n_val + n_train
    n_total_users = df[user_col].nunique()
    user_counts = df.groupby(user_col)[user_col].transform("count")
    df = df[user_counts >= min_required].copy()
    n_excluded = n_total_users - df[user_col].nunique()
    logging.info(
        f"Users excluded (fewer than {min_required} interactions): {n_excluded} / {n_total_users}"
    )

    df = df.sort_values(by=[user_col, time_col, "item_id"], kind="stable").copy()
    df["reverse_rank"] = df.groupby(user_col).cumcount(ascending=False)

    test_mask = df["reverse_rank"] < n_test
    val_mask = (df["reverse_rank"] >= n_test) & (df["reverse_rank"] < n_test + n_val)

    test_df = df[test_mask].drop(columns=["reverse_rank"])
    val_df = df[val_mask].drop(columns=["reverse_rank"])
    train_df = df[~test_mask & ~val_mask].drop(columns=["reverse_rank"])

    return train_df, val_df, test_df, n_excluded


def run_v1_checks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str,
    time_col: str,
):
    """Runs integrity checks on the V1 split.

    Verifies no index overlap between splits, that val/test users are all
    present in train, and that temporal order is respected (train < val < test).

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        user_col (str): Name of the user column.
        time_col (str): Name of the timestamp column.

    Returns:
        None
    """
    assert len(set(train_df.index) & set(val_df.index)) == 0, "Overlap train/val"
    assert len(set(train_df.index) & set(test_df.index)) == 0, "Overlap train/test"
    assert len(set(val_df.index) & set(test_df.index)) == 0, "Overlap val/test"

    train_users = set(train_df[user_col].unique())
    assert set(val_df[user_col].unique()).issubset(train_users), "Val users missing from train"
    assert set(test_df[user_col].unique()).issubset(train_users), "Test users missing from train"

    val_users = list(val_df[user_col].unique())
    max_train_ts = train_df.groupby(user_col)[time_col].max()
    min_val_ts = val_df.groupby(user_col)[time_col].min()
    max_val_ts = val_df.groupby(user_col)[time_col].max()
    min_test_ts = test_df.groupby(user_col)[time_col].min()

    assert (max_train_ts.loc[val_users] <= min_val_ts.loc[val_users]).all(), (
        "Temporal leakage: train > val"
    )
    assert (max_val_ts.loc[val_users] <= min_test_ts.loc[val_users]).all(), (
        "Temporal leakage: val > test"
    )


def calculate_sparsity(df: pd.DataFrame, user_col: str, item_col: str) -> float:
    """Calculates sparsity of an interaction matrix.

    Args:
        df (pd.DataFrame): Interactions dataframe.
        user_col (str): Name of the user column.
        item_col (str): Name of the item column.

    Returns:
        float: Sparsity value between 0 and 1.
    """
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    return 1.0 - (len(df) / (n_users * n_items)) if n_users and n_items else 1.0


def print_split_stats_v1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str,
    item_col: str,
):
    """Logs a summary table of the V1 train/val/test split.

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        user_col (str): Name of the user column.
        item_col (str): Name of the item column.
    """
    stats = f"""
### Split V1 Statistics

| Metric | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| Users | {train_df[user_col].nunique():,} | {val_df[user_col].nunique():,} | {test_df[user_col].nunique():,} |
| Items | {train_df[item_col].nunique():,} | {val_df[item_col].nunique():,} | {test_df[item_col].nunique():,} |
| Interactions | {len(train_df):,} | {len(val_df):,} | {len(test_df):,} |
| Sparsity | {calculate_sparsity(train_df, user_col, item_col):.4%} | {calculate_sparsity(val_df, user_col, item_col):.4%} | {calculate_sparsity(test_df, user_col, item_col):.4%} |
    """
    logging.info(stats)


def build_item_tables(processed_dir: Path):
    """Builds enriched item tables (item_tags, item_text) and runs integrity checks.

    Reads items.parquet and tags.parquet to produce:
    - item_tags.parquet : (item_id, tags) — unique tags per item, space-separated
    - item_text.parquet : (item_id, text) — title + genres + tags concatenated

    Also verifies that all item_ids in interactions.parquet exist in items.parquet.

    Args:
        processed_dir (Path): Path to the processed data directory.
    """
    items_df = pd.read_parquet(processed_dir / "items.parquet")
    tags_df = pd.read_parquet(processed_dir / "tags.parquet")
    interactions_df = pd.read_parquet(processed_dir / "interactions.parquet")

    # --- item_tags.parquet ---
    item_tags_df = (
        tags_df.rename(columns={"movie_id": "item_id"})
        .groupby("item_id")["tag"]
        .apply(lambda x: " ".join(x.dropna().unique()))
        .reset_index()
        .rename(columns={"tag": "tags"})
    )
    item_tags_df.to_parquet(processed_dir / "item_tags.parquet", index=False)
    logging.info(f"item_tags.parquet: {len(item_tags_df):,} items with tags")

    # --- item_text.parquet ---
    text_df = items_df.merge(item_tags_df, on="item_id", how="left")

    # genres: replace pipe separator with space
    text_df["genres_clean"] = text_df["genres"].str.replace("|", " ", regex=False)

    # tags: fallback to empty string if item has no tags
    text_df["tags"] = text_df["tags"].fillna("")

    # text: title + genres + tags (strip to avoid leading/trailing spaces)
    text_df["text"] = (
        text_df["title"].fillna("")
        + " "
        + text_df["genres_clean"].fillna("")
        + " "
        + text_df["tags"]
    ).str.strip()

    item_text_df = text_df[["item_id", "text"]]
    item_text_df.to_parquet(processed_dir / "item_text.parquet", index=False)
    logging.info(f"item_text.parquet: {len(item_text_df):,} items")

    # --- Integrity checks ---
    # 1. No NaN in item_text.text
    n_nan_text = item_text_df["text"].isna().sum()
    assert n_nan_text == 0, f"NaN in item_text.text: {n_nan_text} rows"

    # 2. All item_ids in interactions exist in items
    interaction_ids = set(interactions_df["item_id"].unique())
    item_ids = set(items_df["item_id"].unique())
    missing = interaction_ids - item_ids
    assert len(missing) == 0, f"item_ids in interactions missing from items: {missing}"

    logging.info("Item table integrity checks passed.")


def collect_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_excluded: int,
    user_col: str,
    item_col: str,
) -> dict:
    """Collects V1 split statistics.

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        n_excluded (int): Number of users excluded for having too few interactions.
        user_col (str): Name of the user column.
        item_col (str): Name of the item column.

    Returns:
        dict: Split stats (users_excluded, train/val/test sizes).
    """
    return {
        "users_excluded": n_excluded,
        "train": {
            "users": int(train_df[user_col].nunique()),
            "items": int(train_df[item_col].nunique()),
            "interactions": len(train_df),
        },
        "val": {
            "users": int(val_df[user_col].nunique()),
            "items": int(val_df[item_col].nunique()),
            "interactions": len(val_df),
        },
        "test": {
            "users": int(test_df[user_col].nunique()),
            "items": int(test_df[item_col].nunique()),
            "interactions": len(test_df),
        },
    }


def write_metadata(metadata: dict, processed_dir: Path):
    """Writes the full metadata to data/reports/metadata.json.

    Args:
        metadata (dict): Complete metadata dict to serialize.
        processed_dir (Path): Path to the processed data directory.
    """
    report_dir = processed_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = report_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata written to {metadata_path}")


def process_remaining_files(con, raw_dir, processed_dir, traited_files):
    """Processes remaining CSV files with generic conversion.

    Args:
        con: DuckDB connection.
        raw_dir (Path): Path to raw data directory.
        processed_dir (Path): Path to processed data directory.
        traited_files (list): List of files already processed.
    """
    for input_csv in raw_dir.glob("*.csv"):
        if input_csv.name not in traited_files:
            print(f"Generic conversion of {input_csv.name}...")
            output_pq = processed_dir / input_csv.name.replace(".csv", ".parquet")
            con.execute(
                f"COPY (SELECT * FROM read_csv_auto('{input_csv}')) TO '{output_pq}' (FORMAT PARQUET)"
            )


def main():
    """Main function to process raw MovieLens data into processed Parquet files.

    This function orchestrates the data processing pipeline by calling
    individual functions for each step.

    Args:
        None (uses command line arguments for config path)

    Returns:
        None
    """
    args = parse_arguments()

    # Load configuration and extract parameters
    logging.info(f"Loading configuration from {args.config}...")
    config, rating_threshold, min_user_inter, min_item_inter, n_test, n_val, n_train = (
        load_and_extract_config(args.config)
    )

    log_parameters(rating_threshold, min_user_inter, min_item_inter, n_test, n_val, n_train)

    raw_dir, processed_dir, con = setup_directories()

    filtering_stats = process_ratings(
        con, raw_dir, processed_dir, rating_threshold, min_user_inter, min_item_inter
    )

    process_specific_files(con, raw_dir, processed_dir, SPECIFIC_CONVERSIONS)

    process_remaining_files(con, raw_dir, processed_dir, TRAITED_FILES)

    logging.info("Building item tables (item_tags, item_text)...")
    build_item_tables(processed_dir)

    logging.info("Generating V1 split (train/val/test)...")
    df_filtered = pd.read_parquet(processed_dir / "interactions.parquet")
    train_df, val_df, test_df, n_excluded = generate_v1_split(
        df=df_filtered,
        n_test=n_test,
        n_val=n_val,
        n_train=n_train,
        user_col="user_id",
        time_col="timestamp",
    )

    logging.info("Running V1 split integrity checks...")
    run_v1_checks(train_df, val_df, test_df, user_col="user_id", time_col="timestamp")
    logging.info("All checks passed.")

    logging.info("Saving V1 splits to Parquet...")
    train_df.to_parquet(processed_dir / "interactions_train.parquet", index=False)
    val_df.to_parquet(processed_dir / "interactions_val.parquet", index=False)
    test_df.to_parquet(processed_dir / "interactions_test.parquet", index=False)

    print_split_stats_v1(train_df, val_df, test_df, user_col="user_id", item_col="item_id")
    split_stats = collect_split_stats(
        train_df,
        val_df,
        test_df,
        n_excluded,
        user_col="user_id",
        item_col="item_id",
    )

    metadata = {
        "version": config.get("dataset_version", "dataset_v1"),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "ml-25m",
        "config": {
            "rating_threshold": rating_threshold,
            "min_user_interactions": min_user_inter,
            "min_item_interactions": min_item_inter,
            "n_test": n_test,
            "n_val": n_val,
            "n_train": n_train,
            "method": "temporal_split_per_user",
        },
        "paths": {
            "interactions": "data/processed/interactions.parquet",
            "train": "data/processed/interactions_train.parquet",
            "val": "data/processed/interactions_val.parquet",
            "test": "data/processed/interactions_test.parquet",
            "items": "data/processed/items.parquet",
            "item_tags": "data/processed/item_tags.parquet",
            "item_text": "data/processed/item_text.parquet",
        },
        "filtering": filtering_stats,
        "split": split_stats,
    }
    write_metadata(metadata, processed_dir)


if __name__ == "__main__":
    main()
