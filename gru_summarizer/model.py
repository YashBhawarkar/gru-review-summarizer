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
    if config.encoder_bidirectional:
        encoder_sequence, forward_state, backward_state = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                config.hidden_dim,
                dropout=config.dropout,
                return_sequences=True,
                return_state=True,
            ),
            merge_mode="concat",
            name="encoder_bidirectional",
        )(encoder_embedding)
        joined_state = tf.keras.layers.Concatenate(name="encoder_state_concat")(
            [forward_state, backward_state]
        )
        encoder_state = tf.keras.layers.Dense(
            config.hidden_dim, activation="tanh", name="encoder_state_bridge"
        )(joined_state)
        encoder_attention_sequence = tf.keras.layers.Dense(
            config.hidden_dim, activation="tanh", name="encoder_attention_projection"
        )(encoder_sequence)
    else:
        encoder_sequence, encoder_state = tf.keras.layers.GRU(
            config.hidden_dim,
            dropout=config.dropout,
            return_sequences=True,
            return_state=True,
            name="encoder_gru",
        )(encoder_embedding)
        encoder_attention_sequence = encoder_sequence

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
    if config.use_attention:
        attention_context = tf.keras.layers.AdditiveAttention(name="decoder_attention")(
            [decoder_sequence, encoder_attention_sequence]
        )
        decoder_features = tf.keras.layers.Concatenate(name="decoder_context_concat")(
            [decoder_sequence, attention_context]
        )
    else:
        decoder_features = decoder_sequence
    logits = tf.keras.layers.Dense(decoder_vocab_size, name="token_logits")(decoder_features)
    return tf.keras.Model([encoder_inputs, decoder_inputs], logits, name="gru_review_summarizer")


def compile_for_training(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
    )
