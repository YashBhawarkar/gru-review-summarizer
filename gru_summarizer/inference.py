"""Artifact loading and state-correct autoregressive GRU decoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import tensorflow as tf

from .artifacts import (
    CONFIG_FILE,
    DECODER_TOKENIZER_FILE,
    ENCODER_TOKENIZER_FILE,
    MODEL_FILE,
    verify_manifest,
)
from .config import PreprocessingConfig
from .preprocessing import encode_sources, normalize_text, validate_special_tokens


@dataclass(frozen=True)
class SummarizationResult:
    summary: str
    input_tokens: int
    used_tokens: int
    truncated: bool
    inference_seconds: float


class Summarizer:
    def __init__(
        self,
        model: tf.keras.Model,
        encoder_tokenizer: tf.keras.preprocessing.text.Tokenizer,
        decoder_tokenizer: tf.keras.preprocessing.text.Tokenizer,
        config: PreprocessingConfig,
    ) -> None:
        validate_special_tokens(decoder_tokenizer, config)
        self.model = model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.config = config
        self.encoder_embedding = model.get_layer("encoder_embedding")
        self.encoder_gru = model.get_layer("encoder_gru")
        self.decoder_embedding = model.get_layer("decoder_embedding")
        self.decoder_gru = model.get_layer("decoder_gru")
        self.token_logits = model.get_layer("token_logits")
        self.start_id = decoder_tokenizer.word_index[config.start_token]
        self.end_id = decoder_tokenizer.word_index[config.end_token]
        self.index_word = decoder_tokenizer.index_word

    def _decode_batch(self, clean_texts: Sequence[str]) -> list[str]:
        encoder_inputs = encode_sources(clean_texts, self.encoder_tokenizer, self.config)
        embedded = self.encoder_embedding(encoder_inputs, training=False)
        _, state = self.encoder_gru(embedded, training=False)

        batch_size = len(clean_texts)
        current = tf.fill((batch_size, 1), tf.cast(self.start_id, tf.int32))
        decoded: list[list[str]] = [[] for _ in range(batch_size)]
        finished = np.zeros(batch_size, dtype=bool)

        for _ in range(self.config.max_summary_tokens - 2):
            step_embedding = self.decoder_embedding(current, training=False)
            step_output, state = self.decoder_gru(
                step_embedding, initial_state=state, training=False
            )
            logits = self.token_logits(step_output, training=False).numpy()[:, -1, :]
            logits[:, 0] = -np.inf
            logits[:, self.start_id] = -np.inf
            predicted = np.argmax(logits, axis=-1).astype("int32")

            for row, token_id in enumerate(predicted):
                if finished[row]:
                    continue
                if int(token_id) == self.end_id:
                    finished[row] = True
                    continue
                word = self.index_word.get(int(token_id), self.config.oov_token)
                decoded[row].append("[UNK]" if word == self.config.oov_token else word)
            if finished.all():
                break
            predicted[finished] = self.end_id
            current = tf.convert_to_tensor(predicted[:, None], dtype=tf.int32)

        return [" ".join(words).strip() for words in decoded]

    def summarize_batch(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        clean = [normalize_text(text, self.config.lowercase) for text in texts]
        return self._decode_batch(clean)

    def summarize(self, text: str) -> SummarizationResult:
        clean = normalize_text(text, self.config.lowercase)
        token_count = len(clean.split())
        if token_count == 0:
            raise ValueError("Enter at least one word to summarize.")
        started = perf_counter()
        summary = self._decode_batch([clean])[0]
        elapsed = perf_counter() - started
        return SummarizationResult(
            summary=summary,
            input_tokens=token_count,
            used_tokens=min(token_count, self.config.max_input_tokens),
            truncated=token_count > self.config.max_input_tokens,
            inference_seconds=elapsed,
        )


def load_summarizer(directory: str | Path, verify_checksums: bool = True) -> Summarizer:
    directory = Path(directory)
    if verify_checksums:
        verify_manifest(directory)
    config = PreprocessingConfig.load(directory / CONFIG_FILE)
    encoder_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
        (directory / ENCODER_TOKENIZER_FILE).read_text(encoding="utf-8")
    )
    decoder_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
        (directory / DECODER_TOKENIZER_FILE).read_text(encoding="utf-8")
    )
    model = tf.keras.models.load_model(directory / MODEL_FILE, compile=False)
    return Summarizer(model, encoder_tokenizer, decoder_tokenizer, config)
