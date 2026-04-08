import argparse
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
            CAST(movieId AS INTEGER) AS movie_id,
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
            CAST(movieId AS INTEGER) AS movie_id,
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
        item_counts = df["movie_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_inter].index
        df = df[df["movie_id"].isin(valid_items)]

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
        df_before["movie_id"].nunique(),
        len(df_before),
    )
    users_a, items_a, inter_a = (
        df_after["user_id"].nunique(),
        df_after["movie_id"].nunique(),
        len(df_after),
    )

    pct_dropped = ((inter_b - inter_a) / inter_b) * 100

    # Helper function to compute min, median, and max interactions per user/item
    def get_stats(df, column):
        counts = df[column].value_counts()
        return counts.min(), counts.median(), counts.max()

    u_min_b, u_med_b, u_max_b = get_stats(df_before, "user_id")
    u_min_a, u_med_a, u_max_a = get_stats(df_after, "user_id")

    i_min_b, i_med_b, i_max_b = get_stats(df_before, "movie_id")
    i_min_a, i_med_a, i_max_a = get_stats(df_after, "movie_id")

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


def save_filtering_stats(df_before, df_after, config, output_dir):
    """
    Saves filtering statistics to metadata.json and a markdown report.

    Args:
        df_before (pd.DataFrame): Dataframe before filtering.
        df_after (pd.DataFrame): Dataframe after filtering.
        config (dict): Configuration parameters used for filtering.
        output_dir (Path): Directory where the reports should be saved.

    Returns:
        None
    """
    report_dir = output_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Compile stats into a dictionary
    stats = {
        "config_used": config,
        "results": {
            "users": {
                "before": int(df_before["user_id"].nunique()),
                "after": int(df_after["user_id"].nunique()),
            },
            "items": {
                "before": int(df_before["movie_id"].nunique()),
                "after": int(df_after["movie_id"].nunique()),
            },
            "interactions": {"before": len(df_before), "after": len(df_after)},
            "pct_removed": round(((len(df_before) - len(df_after)) / len(df_before)) * 100, 2),
        },
    }

    # Save stats to metadata.json
    with open(report_dir / "metadata.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Generate a markdown report summarizing the filtering results
    with open(report_dir / "data_cleaning.md", "w") as f:
        f.write("# Data Cleaning Report\n\n")
        f.write("## Configuration\n")
        for k, v in config.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Results\n")
        f.write("| Metric | Before | After | Change |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(
            f"| Users | {stats['results']['users']['before']} | {stats['results']['users']['after']} | - |\n"
        )
        f.write(
            f"| Items | {stats['results']['items']['before']} | {stats['results']['items']['after']} | - |\n"
        )
        f.write(
            f"| Interactions | {stats['results']['interactions']['before']} | {stats['results']['interactions']['after']} | -{stats['results']['pct_removed']}% |\n"
        )

    logging.info(f"Stats saved to {report_dir}")


def process_ratings(con, raw_dir, processed_dir, rating_threshold, min_user_inter, min_item_inter):
    """Processes ratings.csv to create interactions.parquet.

    Args:
        con: DuckDB connection.
        raw_dir (Path): Path to raw data directory.
        processed_dir (Path): Path to processed data directory.
        rating_threshold (float): Minimum rating to consider as interaction.
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
                CAST(movieId AS INTEGER) AS movie_id,
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

        current_config = {
            "rating_threshold": rating_threshold,
            "min_user_interactions": min_user_inter,
            "min_item_interactions": min_item_inter,
        }
        save_filtering_stats(
            df_before=df_raw_interactions,
            df_after=df_filtered_interactions,
            config=current_config,
            output_dir=interactions_pq,
        )
    else:
        print(f"File {ratings_csv} not found")


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

    process_ratings(con, raw_dir, processed_dir, rating_threshold, min_user_inter, min_item_inter)

    process_specific_files(con, raw_dir, processed_dir, SPECIFIC_CONVERSIONS)

    process_remaining_files(con, raw_dir, processed_dir, TRAITED_FILES)


if __name__ == "__main__":
    main()
