"""Serializable configuration shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    artifact_version: str = "1.0"
    max_input_tokens: int = 80
    max_summary_tokens: int = 10
    encoder_vocab_limit: int = 8_000
    decoder_vocab_limit: int = 3_000
    embedding_dim: int = 96
    hidden_dim: int = 160
    dropout: float = 0.10
    lowercase: bool = True
    padding: str = "post"
    truncating: str = "post"
    oov_token: str = "<unk>"
    start_token: str = "sostok"
    end_token: str = "eostok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
