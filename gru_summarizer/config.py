"""Serializable configuration shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    artifact_version: str = "2.0"
    max_input_tokens: int = 120
    max_summary_tokens: int = 12
    encoder_vocab_limit: int = 20_000
    decoder_vocab_limit: int = 6_000
    embedding_dim: int = 128
    hidden_dim: int = 192
    dropout: float = 0.10
    encoder_bidirectional: bool = True
    use_attention: bool = True
    beam_width: int = 2
    length_penalty: float = 1.0
    min_summary_tokens: int = 2
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
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # Artifacts created before v2 used greedy decoding. Preserve that behavior
        # when loading them so old checkpoints remain exactly reproducible.
        if "beam_width" not in payload:
            payload.update(beam_width=1, length_penalty=0.0, min_summary_tokens=0)
        if "encoder_bidirectional" not in payload:
            payload.update(encoder_bidirectional=False, use_attention=False)
        supported = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - supported)
        if unknown:
            raise ValueError(
                "Artifact preprocessing configuration requires newer application code; "
                f"unsupported fields: {', '.join(unknown)}"
            )
        return cls(**payload)
