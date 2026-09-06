"""Download and deterministically sample the CC0-tagged review/title corpus."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_HANDLE = "kritanjalijain/amazon-reviews/versions/2"
SOURCE_ROWS = 3_600_000
CHUNK_SIZE = 100_000
DEFAULT_SHA256 = "4743846b93a68e9295aa1dd3cb1dd982cb7861a838aa7cbf7be7ffd97630ebc0"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/amazon_reviews_cc0_sample.parquet"),
    )
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=120)
    parser.add_argument("--max-title-tokens", type=int, default=10)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/download-cache"),
        help="KaggleHub cache; the 1.29 GiB source archive is not committed",
    )
    args = parser.parse_args()
    if args.dataset_size < 1:
        parser.error("--dataset-size must be a positive integer")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    is_default_sample = (
        args.dataset_size == 120_000
        and args.seed == 42
        and args.max_input_tokens == 120
        and args.max_title_tokens == 10
    )
    if args.output.exists() and is_default_sample and checksum(args.output) == DEFAULT_SHA256:
        print(f"Verified existing deterministic dataset: {args.output} ({DEFAULT_SHA256})")
        return
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(args.cache_dir.resolve()))

    # Import after setting KAGGLEHUB_CACHE so the large source never lands in a
    # user's home directory by surprise. Public dataset downloads need no key.
    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    source = dataset_dir / "train.csv"
    if not source.exists():
        raise FileNotFoundError(f"Kaggle download did not contain train.csv: {dataset_dir}")

    rng = np.random.default_rng(args.seed)
    chunk_count = math.ceil(SOURCE_ROWS / CHUNK_SIZE)
    per_chunk = math.ceil(args.dataset_size / chunk_count * 1.2)
    samples: list[pd.DataFrame] = []
    rows_scanned = 0
    eligible_rows = 0
    for chunk in pd.read_csv(
        source,
        header=None,
        names=["label", "title", "review_text"],
        chunksize=CHUNK_SIZE,
        dtype={"label": "int8", "title": "string", "review_text": "string"},
    ):
        rows_scanned += len(chunk)
        chunk = chunk.dropna(subset=["title", "review_text"])
        source_lengths = chunk["review_text"].str.split().str.len()
        title_lengths = chunk["title"].str.split().str.len()
        chunk = chunk[
            source_lengths.between(4, args.max_input_tokens)
            & title_lengths.between(1, args.max_title_tokens)
        ]
        eligible_rows += len(chunk)
        take = min(per_chunk, len(chunk))
        samples.append(chunk.iloc[rng.choice(len(chunk), size=take, replace=False)])

    sampled = pd.concat(samples, ignore_index=True)
    sampled = sampled.iloc[rng.permutation(len(sampled))]
    # The training pipeline repeats normalized-text deduplication before any
    # split; this raw pass prevents obvious duplicates consuming sample slots.
    sampled = sampled.drop_duplicates(subset=["review_text"], keep="first")
    sampled = sampled.iloc[: args.dataset_size].reset_index(drop=True)
    if len(sampled) < args.dataset_size:
        raise RuntimeError(
            f"Only {len(sampled):,} unique eligible pairs were sampled; "
            "request a smaller --dataset-size"
        )

    temporary = args.output.with_suffix(args.output.suffix + ".part")
    sampled.to_parquet(temporary, index=False)
    actual_checksum = checksum(temporary)
    if is_default_sample and actual_checksum != DEFAULT_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Deterministic dataset checksum mismatch: expected {DEFAULT_SHA256}, "
            f"got {actual_checksum}"
        )
    temporary.replace(args.output)
    print(
        f"Scanned {rows_scanned:,} rows; {eligible_rows:,} met the raw length filters.\n"
        f"Saved {len(sampled):,} deterministic pairs to {args.output}.\n"
        f"SHA-256: {actual_checksum}"
    )


if __name__ == "__main__":
    main()
