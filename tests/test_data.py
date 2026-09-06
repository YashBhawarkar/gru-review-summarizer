import pandas as pd

from gru_summarizer.config import PreprocessingConfig
from gru_summarizer.data import load_reviews, prepare_reviews, split_reviews


def test_loads_downloaded_amazon_column_pair(tmp_path):
    path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "label": [2],
            "title": ["Excellent headphones"],
            "review_text": ["Clear sound and comfortable ear cups."],
        }
    ).to_parquet(path, index=False)
    loaded = load_reviews(path)
    assert loaded.to_dict("records") == [
        {
            "source": "Clear sound and comfortable ear cups.",
            "target": "Excellent headphones",
        }
    ]


def test_deduplication_happens_before_split():
    frame = pd.DataFrame(
        {
            "source": [
                "This dress fits very well",
                "This dress fits very well!",
                "The zipper broke immediately",
                "Soft fabric and lovely color",
                "Runs small but looks beautiful",
                "The sleeves are much too long",
                "Perfect for a summer wedding",
                "Material feels thin and cheap",
                "I would buy another one",
                "Comfortable enough for all day",
            ],
            "target": ["great fit", "duplicate", "broken zipper", "soft and lovely", "runs small", "long sleeves", "summer dress", "thin material", "would rebuy", "all day comfort"],
        }
    )
    prepared, stats = prepare_reviews(frame, PreprocessingConfig(), None, 7)
    assert stats["duplicates_removed_before_split"] == 1
    train, validation, test = split_reviews(prepared, 0.6, 0.2)
    assert len(train) + len(validation) + len(test) == len(prepared)
    assert len(set(train.source) & set(test.source)) == 0
