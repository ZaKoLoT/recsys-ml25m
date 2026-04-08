import argparse
import logging
from pathlib import Path

import duckdb
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
        tuple: (project_dir, raw_dir, processed_dir, con)
    """
    project_dir = Path(__file__).resolve().parent.parent
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    return project_dir, raw_dir, processed_dir, con


def process_ratings(con, raw_dir, processed_dir, rating_threshold):
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
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration and extract parameters
    logging.info(f"Loading configuration from {args.config}...")
    config, rating_threshold, min_user_inter, min_item_inter, n_test, n_val, n_train = (
        load_and_extract_config(args.config)
    )

    # Log the used parameters
    log_parameters(rating_threshold, min_user_inter, min_item_inter, n_test, n_val, n_train)

    # Set up directories and database connection
    project_dir, raw_dir, processed_dir, con = setup_directories()

    # Process ratings.csv
    process_ratings(con, raw_dir, processed_dir, rating_threshold)

    # Process specific files
    process_specific_files(con, raw_dir, processed_dir, SPECIFIC_CONVERSIONS)

    # Process remaining files
    process_remaining_files(con, raw_dir, processed_dir, TRAITED_FILES)


if __name__ == "__main__":
    main()
