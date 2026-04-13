import pytest

from src.recsys.data import load_items, load_splits, load_tags, load_text


@pytest.fixture(scope="module")
def splits():
    return load_splits()


@pytest.fixture(scope="module")
def items():
    return load_items()


def test_splits_load(splits):
    train, val, test = splits
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_splits_columns(splits):
    expected = {"user_id", "movie_id", "timestamp", "interaction"}
    train, val, test = splits
    for df in (train, val, test):
        assert expected.issubset(set(df.columns))


def test_items_load(items):
    assert len(items) > 0


def test_items_columns(items):
    expected = {"item_id", "title", "genres"}
    assert expected.issubset(set(items.columns))


def test_tags_load():
    tags = load_tags()
    assert len(tags) > 0
    assert {"movie_id", "tags"}.issubset(set(tags.columns))


def test_text_load():
    text = load_text()
    assert len(text) > 0
    assert {"movie_id", "text"}.issubset(set(text.columns))
    assert text["text"].isna().sum() == 0


def test_referential_integrity(splits, items):
    train, val, test = splits
    interaction_ids = (
        set(train["movie_id"].unique())
        | set(val["movie_id"].unique())
        | set(test["movie_id"].unique())
    )
    item_ids = set(items["item_id"].unique())
    missing = interaction_ids - item_ids
    assert len(missing) == 0, f"{len(missing)} movie_ids in splits missing from items"
