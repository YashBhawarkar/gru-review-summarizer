"""Dataset loading, global deduplication, filtering, and deterministic splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PreprocessingConfig
from .preprocessing import normalize_text


_COLUMN_PAIRS = (
    ("review_text", "title"),
    ("Review Text", "Title"),
    ("Text", "Summary"),
    ("review_body", "review_title"),
)


def load_reviews(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    for source_col, target_col in _COLUMN_PAIRS:
        if source_col in frame.columns and target_col in frame.columns:
            return frame[[source_col, target_col]].rename(
                columns={source_col: "source", target_col: "target"}
            )
    raise ValueError(
        "Dataset needs one of these source/summary column pairs: "
        + ", ".join(f"{a}/{b}" for a, b in _COLUMN_PAIRS)
    )


def prepare_reviews(
    frame: pd.DataFrame,
    config: PreprocessingConfig,
    dataset_size: int | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean and deduplicate the complete pool before sampling or splitting."""
    original_count = len(frame)
    clean = frame.dropna(subset=["source", "target"]).copy()
    clean["source"] = clean["source"].map(lambda x: normalize_text(x, config.lowercase))
    clean["target"] = clean["target"].map(lambda x: normalize_text(x, config.lowercase))
    clean = clean[(clean["source"] != "") & (clean["target"] != "")]

    before_dedup = len(clean)
    clean = clean.drop_duplicates(subset=["source"], keep="first")
    duplicates_removed = before_dedup - len(clean)

    source_lengths = clean["source"].str.split().str.len()
    target_lengths = clean["target"].str.split().str.len()
    clean = clean[
        source_lengths.between(4, config.max_input_tokens)
        & target_lengths.between(1, config.max_summary_tokens - 2)
    ].reset_index(drop=True)
    after_filtering = len(clean)

    rng = np.random.default_rng(seed)
    clean = clean.iloc[rng.permutation(len(clean))].reset_index(drop=True)
    if dataset_size is not None and dataset_size > 0:
        clean = clean.iloc[: min(dataset_size, len(clean))].copy()

    stats = {
        "raw_rows": original_count,
        "complete_rows_before_deduplication": before_dedup,
        "duplicates_removed_before_split": duplicates_removed,
        "eligible_rows_after_length_filtering": after_filtering,
        "selected_rows": len(clean),
        "unique_normalized_titles": int(clean["target"].nunique()),
    }
    return clean, stats


def split_reviews(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be less than one")
    train_end = int(len(frame) * train_fraction)
    validation_end = train_end + int(len(frame) * validation_fraction)
    train = frame.iloc[:train_end].reset_index(drop=True)
    validation = frame.iloc[train_end:validation_end].reset_index(drop=True)
    test = frame.iloc[validation_end:].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Dataset is too small for non-empty train/validation/test splits")
    return train, validation, test
