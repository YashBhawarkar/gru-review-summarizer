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
        self.decoder_embedding = model.get_layer("decoder_embedding")
        self.decoder_gru = model.get_layer("decoder_gru")
        self.token_logits = model.get_layer("token_logits")
        self.is_bidirectional = any(layer.name == "encoder_bidirectional" for layer in model.layers)
        self.uses_attention = any(layer.name == "decoder_attention" for layer in model.layers)
        if self.is_bidirectional:
            self.encoder_recurrent = model.get_layer("encoder_bidirectional")
            self.encoder_state_concat = model.get_layer("encoder_state_concat")
            self.encoder_state_bridge = model.get_layer("encoder_state_bridge")
            self.encoder_attention_projection = model.get_layer("encoder_attention_projection")
        else:
            self.encoder_recurrent = model.get_layer("encoder_gru")
        if self.uses_attention:
            self.decoder_attention = model.get_layer("decoder_attention")
            self.decoder_context_concat = model.get_layer("decoder_context_concat")
        self.start_id = decoder_tokenizer.word_index[config.start_token]
        self.end_id = decoder_tokenizer.word_index[config.end_token]
        self.index_word = decoder_tokenizer.index_word

    def _encode(self, clean_texts: Sequence[str]):
        encoder_inputs = encode_sources(clean_texts, self.encoder_tokenizer, self.config)
        embedded = self.encoder_embedding(encoder_inputs, training=False)
        encoder_mask = tf.not_equal(encoder_inputs, 0)
        if self.is_bidirectional:
            encoder_sequence, forward_state, backward_state = self.encoder_recurrent(
                embedded, mask=encoder_mask, training=False
            )
            joined_state = self.encoder_state_concat([forward_state, backward_state])
            state = self.encoder_state_bridge(joined_state, training=False)
            encoder_sequence = self.encoder_attention_projection(encoder_sequence, training=False)
        else:
            encoder_outputs = self.encoder_recurrent(
                embedded, mask=encoder_mask, training=False
            )
            if self.encoder_recurrent.return_sequences:
                encoder_sequence, state = encoder_outputs
            else:
                _, state = encoder_outputs
                encoder_sequence = None

        return state, encoder_sequence, encoder_mask

    def _decode_batch(
        self,
        clean_texts: Sequence[str],
        beam_width: int | None = None,
        length_penalty: float | None = None,
        min_summary_tokens: int | None = None,
    ) -> list[str]:
        beam_width = self.config.beam_width if beam_width is None else beam_width
        length_penalty = (
            self.config.length_penalty if length_penalty is None else length_penalty
        )
        min_summary_tokens = (
            self.config.min_summary_tokens
            if min_summary_tokens is None
            else min_summary_tokens
        )
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if length_penalty < 0:
            raise ValueError("length_penalty must be non-negative")
        if not 0 <= min_summary_tokens <= self.config.max_summary_tokens - 2:
            raise ValueError("min_summary_tokens is outside the configured summary length")

        state, encoder_sequence, encoder_mask = self._encode(clean_texts)
        if beam_width == 1 and length_penalty == 0 and min_summary_tokens == 0:
            return self._decode_greedy(clean_texts, state, encoder_sequence, encoder_mask)
        return self._decode_beam(
            clean_texts,
            state,
            encoder_sequence,
            encoder_mask,
            beam_width,
            length_penalty,
            min_summary_tokens,
        )

    def _decode_greedy(self, clean_texts, state, encoder_sequence, encoder_mask):

        batch_size = len(clean_texts)
        current = tf.fill((batch_size, 1), tf.cast(self.start_id, tf.int32))
        decoded: list[list[str]] = [[] for _ in range(batch_size)]
        finished = np.zeros(batch_size, dtype=bool)

        for _ in range(self.config.max_summary_tokens - 2):
            step_embedding = self.decoder_embedding(current, training=False)
            step_output, state = self.decoder_gru(
                step_embedding, initial_state=state, training=False
            )
            if self.uses_attention:
                query_mask = tf.ones((batch_size, 1), dtype=tf.bool)
                context = self.decoder_attention(
                    [step_output, encoder_sequence],
                    mask=[query_mask, encoder_mask],
                    training=False,
                )
                decoder_features = self.decoder_context_concat([step_output, context])
            else:
                decoder_features = step_output
            logits = self.token_logits(decoder_features, training=False).numpy()[:, -1, :]
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

    def _decode_beam(
        self,
        clean_texts,
        initial_state,
        encoder_sequence,
        encoder_mask,
        beam_width,
        length_penalty,
        min_summary_tokens,
    ):
        """Batched, length-normalized beam search over the trained decoder."""
        batch_size = len(clean_texts)
        max_steps = self.config.max_summary_tokens - 2
        vocab_size = self.token_logits.units

        state = np.repeat(initial_state.numpy()[:, None, :], beam_width, axis=1)
        current = np.full((batch_size, beam_width), self.start_id, dtype="int32")
        scores = np.full((batch_size, beam_width), -np.inf, dtype="float32")
        scores[:, 0] = 0.0
        finished = np.zeros((batch_size, beam_width), dtype=bool)
        lengths = np.zeros((batch_size, beam_width), dtype="int32")
        histories = np.zeros((batch_size, beam_width, max_steps), dtype="int32")
        batch_rows = np.arange(batch_size)[:, None]

        if self.uses_attention:
            repeated_sequence = tf.repeat(encoder_sequence, repeats=beam_width, axis=0)
            repeated_mask = tf.repeat(encoder_mask, repeats=beam_width, axis=0)

        for step in range(max_steps):
            flat_current = tf.convert_to_tensor(current.reshape(-1, 1), dtype=tf.int32)
            flat_state = tf.convert_to_tensor(
                state.reshape(batch_size * beam_width, -1), dtype=tf.float32
            )
            step_embedding = self.decoder_embedding(flat_current, training=False)
            step_output, next_state = self.decoder_gru(
                step_embedding, initial_state=flat_state, training=False
            )
            if self.uses_attention:
                query_mask = tf.ones((batch_size * beam_width, 1), dtype=tf.bool)
                context = self.decoder_attention(
                    [step_output, repeated_sequence],
                    mask=[query_mask, repeated_mask],
                    training=False,
                )
                decoder_features = self.decoder_context_concat([step_output, context])
            else:
                decoder_features = step_output

            token_scores = tf.nn.log_softmax(
                self.token_logits(decoder_features, training=False)[:, -1, :], axis=-1
            ).numpy().reshape(batch_size, beam_width, vocab_size)
            token_scores[:, :, 0] = -np.inf
            token_scores[:, :, self.start_id] = -np.inf
            if step < min_summary_tokens:
                token_scores[:, :, self.end_id] = -np.inf

            # A finished beam can only emit another zero-cost end token, which
            # preserves its cumulative score while other beams continue.
            finished_scores = np.full_like(token_scores, -np.inf)
            finished_scores[:, :, self.end_id] = 0.0
            token_scores = np.where(finished[:, :, None], finished_scores, token_scores)
            candidates = scores[:, :, None] + token_scores
            flat_candidates = candidates.reshape(batch_size, -1)
            top = np.argpartition(flat_candidates, -beam_width, axis=1)[:, -beam_width:]
            top_values = np.take_along_axis(flat_candidates, top, axis=1)
            order = np.argsort(top_values, axis=1)[:, ::-1]
            top = np.take_along_axis(top, order, axis=1)
            scores = np.take_along_axis(flat_candidates, top, axis=1)
            parents = top // vocab_size
            tokens = (top % vocab_size).astype("int32")

            parent_finished = finished[batch_rows, parents]
            parent_lengths = lengths[batch_rows, parents]
            histories = histories[batch_rows, parents]
            histories[:, :, step] = tokens
            next_state = next_state.numpy().reshape(batch_size, beam_width, -1)
            state = next_state[batch_rows, parents]
            lengths = parent_lengths + (~parent_finished & (tokens != self.end_id))
            finished = parent_finished | (tokens == self.end_id)
            current = tokens
            if finished.all():
                break

        penalties = np.power((5.0 + np.maximum(lengths, 1)) / 6.0, length_penalty)
        best = np.argmax(scores / penalties, axis=1)
        decoded = []
        for row, beam in enumerate(best):
            words = []
            for token_id in histories[row, beam]:
                if token_id in (0, self.end_id):
                    break
                word = self.index_word.get(int(token_id), self.config.oov_token)
                words.append("[UNK]" if word == self.config.oov_token else word)
            decoded.append(" ".join(words).strip())
        return decoded

    def summarize_batch(
        self,
        texts: Sequence[str],
        beam_width: int | None = None,
        length_penalty: float | None = None,
        min_summary_tokens: int | None = None,
    ) -> list[str]:
        if not texts:
            return []
        clean = [normalize_text(text, self.config.lowercase) for text in texts]
        return self._decode_batch(clean, beam_width, length_penalty, min_summary_tokens)

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
