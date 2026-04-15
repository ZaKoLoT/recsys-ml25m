import logging

from src.recsys.data import load_splits
from src.recsys.evaluation import evaluate_model, write_report
from src.recsys.models.popularity import fit_popularity, recommend

logging.basicConfig(level=logging.INFO, format="%(message)s")

K = 10


def main():
    logging.info("Loading splits...")
    train, val, test = load_splits()

    logging.info("Fitting popularity on train only...")
    ranking = fit_popularity(train)

    logging.info("Building seen items per user from train...")
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    recommend_fn = lambda _, seen, k: recommend(ranking, seen, k)  # noqa: E731

    logging.info("Evaluating on val...")
    val_results = evaluate_model(val, train_seen, recommend_fn, k=K)
    logging.info(f"Val  -> {val_results}")

    logging.info("Evaluating on test...")
    test_results = evaluate_model(test, train_seen, recommend_fn, k=K)
    logging.info(f"Test -> {test_results}")

    write_report(
        model_name="baseline_popularity",
        val_results=val_results,
        test_results=test_results,
        k=K,
        params={
            "K": K,
            "method": "global item interaction count on train",
            "leakage": "train items excluded from recommendations",
        },
    )


if __name__ == "__main__":
    main()
