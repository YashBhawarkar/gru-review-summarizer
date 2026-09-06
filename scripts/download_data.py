"""Download the pinned CC0 review/title dataset and verify its checksum."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request


REVISION = "81458f32611e8f8d78539b5d44cd4c2dc2c98000"
URL = (
    "https://huggingface.co/datasets/chibifire/"
    f"kaggle-womens-ecom-clothing-reviews/resolve/{REVISION}/"
    "data/train-00000-of-00001.parquet"
)
SHA256 = "2350fc698612b568149115425014a94565ea46daa5d3e6136989876d8f4a1637"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/womens_clothing_reviews.parquet"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and checksum(args.output) == SHA256:
        print(f"Verified existing dataset: {args.output}")
        return
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    print(f"Downloading pinned CC0 dataset revision {REVISION}...")
    with urllib.request.urlopen(URL) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target)
    actual = checksum(temporary)
    if actual != SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Dataset checksum mismatch: expected {SHA256}, got {actual}")
    temporary.replace(args.output)
    print(f"Saved and verified {args.output} ({actual})")


if __name__ == "__main__":
    main()
