"""Keras GRU encoder-decoder with teacher forcing."""

from __future__ import annotations

import tensorflow as tf

from .config import PreprocessingConfig


def build_seq2seq_model(
    config: PreprocessingConfig,
    encoder_vocab_size: int,
    decoder_vocab_size: int,
) -> tf.keras.Model:
    encoder_inputs = tf.keras.Input(
        shape=(config.max_input_tokens,), dtype="int32", name="encoder_inputs"
    )
    encoder_embedding = tf.keras.layers.Embedding(
        encoder_vocab_size,
        config.embedding_dim,
        mask_zero=True,
        name="encoder_embedding",
    )(encoder_inputs)
    _, encoder_state = tf.keras.layers.GRU(
        config.hidden_dim,
        dropout=config.dropout,
        return_state=True,
        name="encoder_gru",
    )(encoder_embedding)

    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    decoder_embedding = tf.keras.layers.Embedding(
        decoder_vocab_size,
        config.embedding_dim,
        mask_zero=True,
        name="decoder_embedding",
    )(decoder_inputs)
    decoder_sequence, _ = tf.keras.layers.GRU(
        config.hidden_dim,
        dropout=config.dropout,
        return_sequences=True,
        return_state=True,
        name="decoder_gru",
    )(decoder_embedding, initial_state=encoder_state)
    logits = tf.keras.layers.Dense(decoder_vocab_size, name="token_logits")(decoder_sequence)
    return tf.keras.Model([encoder_inputs, decoder_inputs], logits, name="gru_review_summarizer")


def compile_for_training(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
    )
