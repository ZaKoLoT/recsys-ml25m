# Hybrid and Responsible Recommender System Project

## Project Objective

This project aims to design a **recommender system** in Python (covering movies, series, books, or educational content) that combines:

- A **collaborative** engine, based on interaction history,
- A **content-based** engine, based on descriptions and synopses,
- An **explanation** module ("Why this recommendation?"),
- And a **responsible** module (diversity, novelty, reduction of popularity bias).

---

## Dataset and Volume

- **Main Data:** We use the **MovieLens 25M** dataset, which is the industry standard for this type of exercise.
- **Enrichment:** The data will be paired with the **TMDB (The Movie Database)** API to retrieve full synopses and movie posters, which is essential for semantic (content-based) recommendation.

---

## Deliverables

Upon completion of this project, the following elements will be produced:

- A structured Git repository (`src/`, `notebooks/`, `data/`, `app/`, `tests/`).
- Clean exploratory data analysis (EDA), training, and evaluation notebooks.
- An interactive demonstration application developed with **Streamlit**.
- A 6 to 10-page summary report presenting the methodology, results, limitations, and areas for improvement.

---

## How to Reproduce Data Ingestion

The project is designed to be fully reproducible. To download, clean, and prepare the raw dataset into an optimized format (Parquet), run the following command at the project root:

On Windows: `python scripts/make_dataset.py`
On Mac/Linux: `python3 scripts/make_dataset.py`

To run the full pipeline with the final configuration (iterative cleaning, table creation, and train/val/test split), use:

On Windows: `python scripts/make_dataset.py --config configs/dataset_v1.yaml`
On Mac/Linux: `python3 scripts/make_dataset.py --config configs/dataset_v1.yaml`

---

## **Installation and Development Environment**

If you wish to contribute to the code or run tests, here is how to configure your environment:

- **Clone the project:**
`git clone https://github.com/ZaKoLoT/recsys-ml25m.git`
- **Create and activate the virtual environment:**
On Mac/Linux: `python3 -m venv .venv && source .venv/bin/activate`
On Windows: `python -m venv .venv && .venv\Scripts\activate`
- **Install development dependencies:**
`.\.venv\scripts\python.exe -m pip install -r requirements-dev.txt`
- **Activate automatic checks (pre-commit):**
`pre-commit install`

## Quality Tools

The project follows strict professional standards:

- **Unit Tests:** `pytest -q`
- **Linting (Ruff):** `ruff check .`
- **Formatting (Ruff):** `ruff format .`
- **Manual Pre-commit:** `pre-commit run --all-files`
