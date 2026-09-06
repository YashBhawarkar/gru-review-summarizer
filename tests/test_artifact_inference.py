from pathlib import Path

import pytest

from gru_summarizer.inference import load_summarizer
from gru_summarizer.preprocessing import effective_vocab_size


ARTIFACTS = Path("artifacts")


@pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="trained artifacts not present")
def test_trained_model_reloads_and_generates_autoregressively():
    summarizer = load_summarizer(ARTIFACTS)
    encoder_vocabulary_before = dict(summarizer.encoder_tokenizer.word_index)
    result = summarizer.summarize(
        "The fabric is soft and comfortable, and the dress fits perfectly for an evening out."
    )
    assert result.summary
    assert result.input_tokens > 0
    assert result.used_tokens <= summarizer.config.max_input_tokens
    assert result.inference_seconds >= 0
    assert summarizer.config.start_token not in result.summary
    assert summarizer.config.end_token not in result.summary
    assert summarizer.encoder_tokenizer.word_index == encoder_vocabulary_before
    assert summarizer.model.get_layer("token_logits").units == effective_vocab_size(
        summarizer.decoder_tokenizer
    )
