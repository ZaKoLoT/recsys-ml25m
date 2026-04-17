import logging

from src.recsys.data import load_splits
from src.recsys.evaluation import evaluate_model, write_report
from src.recsys.models.popularity import fit_popularity, recommend

logging.basicConfig(level=logging.INFO, format="%(message)s")

KS = [10, 20]


def main():
    logging.info("Loading splits...")
    train, val, test = load_splits()

    logging.info("Fitting popularity on train only...")
    ranking = fit_popularity(train)

    logging.info("Building seen items per user from train...")
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    recommend_fn = lambda _, seen, k: recommend(ranking, seen, k)  # noqa: E731

    val_results: dict = {}
    test_results: dict = {}

    for k in KS:
        logging.info(f"Evaluating on val with K={k}...")
        results = evaluate_model(val, train_seen, recommend_fn, k=k)
        val_results.update(results)
        logging.info(f"Val K={k} -> {results}")

        logging.info(f"Evaluating on test with K={k}...")
        results = evaluate_model(test, train_seen, recommend_fn, k=k)
        test_results.update(results)
        logging.info(f"Test K={k} -> {results}")

    write_report(
        model_name="baseline_popularity",
        val_results=val_results,
        test_results=test_results,
        ks=KS,
        params={
            "K": ", ".join(str(k) for k in KS),
            "method": "global item interaction count on train",
            "leakage": "train items excluded from recommendations",
        },
    )


if __name__ == "__main__":
    main()
