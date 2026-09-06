"""Deterministic text cleaning, tokenization, and tensor preparation."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from .config import PreprocessingConfig


_NON_WORD_RE = re.compile(r"[^\w']+", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: object, lowercase: bool = True) -> str:
    """Normalize user and dataset text identically without deleting apostrophes."""
    value = html.unescape(unicodedata.normalize("NFKC", str(text)))
    value = value.translate(str.maketrans({"’": "'", "‘": "'", "`": "'"}))
    if lowercase:
        value = value.lower()
    value = value.replace("_", " ")
    value = _NON_WORD_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip(" '")


def add_boundaries(summary: str, config: PreprocessingConfig) -> str:
    clean = normalize_text(summary, config.lowercase)
    return f"{config.start_token} {clean} {config.end_token}".strip()


def make_tokenizers(
    train_texts: Sequence[str],
    train_summaries: Sequence[str],
    config: PreprocessingConfig,
) -> tuple[tf.keras.preprocessing.text.Tokenizer, tf.keras.preprocessing.text.Tokenizer]:
    """Fit both vocabularies on training data only."""
    encoder_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=config.encoder_vocab_limit,
        oov_token=config.oov_token,
        filters="",
        lower=False,
    )
    decoder_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=config.decoder_vocab_limit,
        oov_token=config.oov_token,
        filters="",
        lower=False,
    )
    encoder_tokenizer.fit_on_texts(list(train_texts))
    decoder_tokenizer.fit_on_texts([add_boundaries(item, config) for item in train_summaries])
    return encoder_tokenizer, decoder_tokenizer


def effective_vocab_size(tokenizer: tf.keras.preprocessing.text.Tokenizer) -> int:
    full_size = len(tokenizer.word_index) + 1
    return min(full_size, tokenizer.num_words) if tokenizer.num_words else full_size


def encode_sources(
    texts: Sequence[str],
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    config: PreprocessingConfig,
) -> np.ndarray:
    clean = [normalize_text(item, config.lowercase) for item in texts]
    sequences = tokenizer.texts_to_sequences(clean)
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=config.max_input_tokens,
        padding=config.padding,
        truncating=config.truncating,
        dtype="int32",
    )


def encode_training_pairs(
    texts: Sequence[str],
    summaries: Sequence[str],
    encoder_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    decoder_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    encoder_inputs = encode_sources(texts, encoder_tokenizer, config)
    bounded = [add_boundaries(item, config) for item in summaries]
    summary_sequences = decoder_tokenizer.texts_to_sequences(bounded)
    summary_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        summary_sequences,
        maxlen=config.max_summary_tokens,
        padding=config.padding,
        truncating=config.truncating,
        dtype="int32",
    )
    decoder_inputs = summary_tensor[:, :-1]
    decoder_targets = summary_tensor[:, 1:]
    sample_weights = (decoder_targets != 0).astype("float32")
    return encoder_inputs, decoder_inputs, decoder_targets, sample_weights


def validate_special_tokens(
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    config: PreprocessingConfig,
) -> None:
    missing = [
        token
        for token in (config.oov_token, config.start_token, config.end_token)
        if token not in tokenizer.word_index
    ]
    if missing:
        raise ValueError(f"Decoder tokenizer is missing required tokens: {missing}")
    for token in (config.start_token, config.end_token):
        if tokenizer.word_index[token] >= effective_vocab_size(tokenizer):
            raise ValueError(f"Required token {token!r} falls outside decoder vocabulary limit")
