"""Versioned, checksummed persistence for model and preprocessing artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from .config import PreprocessingConfig


MODEL_FILE = "model.keras"
CONFIG_FILE = "preprocessing.json"
ENCODER_TOKENIZER_FILE = "encoder_tokenizer.json"
DECODER_TOKENIZER_FILE = "decoder_tokenizer.json"
MANIFEST_FILE = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_core_artifacts(
    directory: str | Path,
    model: tf.keras.Model,
    config: PreprocessingConfig,
    encoder_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    decoder_tokenizer: tf.keras.preprocessing.text.Tokenizer,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config.save(directory / CONFIG_FILE)
    (directory / ENCODER_TOKENIZER_FILE).write_text(
        encoder_tokenizer.to_json() + "\n", encoding="utf-8"
    )
    (directory / DECODER_TOKENIZER_FILE).write_text(
        decoder_tokenizer.to_json() + "\n", encoding="utf-8"
    )
    # Saving a fresh, uncompiled model avoids shipping optimizer state to the app.
    inference_model = build_uncompiled_copy(model, config, encoder_tokenizer, decoder_tokenizer)
    inference_model.save(directory / MODEL_FILE)


def build_uncompiled_copy(model, config, encoder_tokenizer, decoder_tokenizer):
    from .model import build_seq2seq_model
    from .preprocessing import effective_vocab_size

    copy = build_seq2seq_model(
        config,
        effective_vocab_size(encoder_tokenizer),
        effective_vocab_size(decoder_tokenizer),
    )
    copy.set_weights(model.get_weights())
    return copy


def write_manifest(directory: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    config_path = directory / CONFIG_FILE
    artifact_version = (
        PreprocessingConfig.load(config_path).artifact_version
        if config_path.exists()
        else "1.0"
    )
    core_files = [MODEL_FILE, CONFIG_FILE, ENCODER_TOKENIZER_FILE, DECODER_TOKENIZER_FILE]
    for optional in ("evaluation.json", "training_history.json"):
        if (directory / optional).exists():
            core_files.append(optional)
    manifest = {
        "artifact_version": artifact_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: _sha256(directory / name) for name in core_files},
        "metadata": metadata,
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_manifest(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"Artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.get("files", {}).items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
    return manifest
