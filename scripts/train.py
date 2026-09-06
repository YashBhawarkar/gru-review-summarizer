"""Train the GRU encoder-decoder; never imported or executed by the hosted app."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from gru_summarizer.artifacts import save_core_artifacts, write_manifest
from gru_summarizer.config import PreprocessingConfig
from gru_summarizer.data import load_reviews, prepare_reviews, split_reviews
from gru_summarizer.model import build_seq2seq_model, compile_for_training
from gru_summarizer.preprocessing import (
    effective_vocab_size,
    encode_training_pairs,
    make_tokenizers,
    validate_special_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/womens_clothing_reviews.parquet"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset-size", type=int, default=0, help="0 uses all eligible rows")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=110)
    parser.add_argument("--max-summary-tokens", type=int, default=12)
    parser.add_argument("--encoder-vocab", type=int, default=10_000)
    parser.add_argument("--decoder-vocab", type=int, default=3_500)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--length-penalty", type=float, default=1.5)
    parser.add_argument("--min-summary-tokens", type=int, default=2)
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def tensors(frame, encoder_tokenizer, decoder_tokenizer, config):
    return encode_training_pairs(
        frame["source"].tolist(),
        frame["target"].tolist(),
        encoder_tokenizer,
        decoder_tokenizer,
        config,
    )


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    config = PreprocessingConfig(
        max_input_tokens=args.max_input_tokens,
        max_summary_tokens=args.max_summary_tokens,
        encoder_vocab_limit=args.encoder_vocab,
        decoder_vocab_limit=args.decoder_vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        encoder_bidirectional=True,
        use_attention=True,
        beam_width=args.beam_width,
        length_penalty=args.length_penalty,
        min_summary_tokens=args.min_summary_tokens,
    )
    frame, data_stats = prepare_reviews(
        load_reviews(args.data), config, args.dataset_size or None, args.seed
    )
    train, validation, test = split_reviews(frame)
    print(f"Rows — train: {len(train)}, validation: {len(validation)}, test: {len(test)}")
    print(json.dumps(data_stats, indent=2))

    # The only tokenizer fit calls in the project occur here, on the train split.
    encoder_tokenizer, decoder_tokenizer = make_tokenizers(
        train["source"].tolist(), train["target"].tolist(), config
    )
    validate_special_tokens(decoder_tokenizer, config)
    train_arrays = tensors(train, encoder_tokenizer, decoder_tokenizer, config)
    validation_arrays = tensors(validation, encoder_tokenizer, decoder_tokenizer, config)

    model = build_seq2seq_model(
        config,
        effective_vocab_size(encoder_tokenizer),
        effective_vocab_size(decoder_tokenizer),
    )
    compile_for_training(model, args.learning_rate)
    model.summary()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(args.processed_dir / "test.csv", index=False)
    checkpoint = args.artifacts_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, args.patience // 2), min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(args.artifacts_dir / "training_log.csv"),
    ]
    history = model.fit(
        [train_arrays[0], train_arrays[1]],
        train_arrays[2],
        sample_weight=train_arrays[3],
        validation_data=(
            [validation_arrays[0], validation_arrays[1]],
            validation_arrays[2],
            validation_arrays[3],
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    history_payload = {key: [float(x) for x in values] for key, values in history.history.items()}
    (args.artifacts_dir / "training_history.json").write_text(
        json.dumps(history_payload, indent=2) + "\n", encoding="utf-8"
    )
    save_core_artifacts(args.artifacts_dir, model, config, encoder_tokenizer, decoder_tokenizer)
    metadata = {
        "trained": True,
        "dataset": "Women's E-Commerce Clothing Reviews (CC0-1.0)",
        "dataset_sha256": "2350fc698612b568149115425014a94565ea46daa5d3e6136989876d8f4a1637",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": args.batch_size,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "data_preparation": data_stats,
        "architecture": {
            "encoder": "bidirectional_gru",
            "decoder": "gru_with_additive_attention",
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "encoder_vocab_size": effective_vocab_size(encoder_tokenizer),
            "decoder_vocab_size": effective_vocab_size(decoder_tokenizer),
            "max_input_tokens": config.max_input_tokens,
            "max_generated_tokens": config.max_summary_tokens - 2,
            "parameter_count": model.count_params(),
            "beam_width": config.beam_width,
            "length_penalty": config.length_penalty,
            "min_summary_tokens": config.min_summary_tokens,
        },
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
    }
    write_manifest(args.artifacts_dir, metadata)
    checkpoint.unlink(missing_ok=True)
    print(f"Saved trained, checksummed artifacts to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
