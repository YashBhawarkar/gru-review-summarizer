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
    assert "[UNK]" not in result.summary
    assert summarizer.encoder_tokenizer.word_index == encoder_vocabulary_before
    assert summarizer.model.get_layer("token_logits").units == effective_vocab_size(
        summarizer.decoder_tokenizer
    )
    assert summarizer.config.artifact_version == "2.0"
    assert summarizer.config.max_input_tokens == 120
    assert summarizer.config.encoder_vocab_limit == 20_000
    assert summarizer.config.decoder_vocab_limit == 6_000
    assert summarizer.config.beam_width == 2
    assert summarizer.is_bidirectional
    assert summarizer.uses_attention
    assert summarizer.model.get_layer("encoder_bidirectional")
    assert summarizer.model.get_layer("decoder_attention")


@pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="trained artifacts not present")
def test_input_contract_reports_truncation_at_saved_limit():
    summarizer = load_summarizer(ARTIFACTS)
    result = summarizer.summarize("comfortable " * 125)
    assert result.input_tokens == 125
    assert result.used_tokens == 120
    assert result.truncated is True


@pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="trained artifacts not present")
def test_reported_cross_domain_failures_no_longer_collapse_to_one_title():
    summarizer = load_summarizer(ARTIFACTS)
    summaries = summarizer.summarize_batch(
        [
            (
                "The sound quality is amazing with deep bass and clear highs. However, the "
                "battery barely lasts three hours and the ear cups become uncomfortable."
            ),
            (
                "Absolutely incredible dining experience. The steak and pasta were cooked to "
                "perfection, the waiter was attentive, and I will definitely return."
            ),
            (
                "The latest update broke the app. It crashes whenever I open a project, and "
                "reinstalling or clearing the cache does not work."
            ),
        ]
    )
    assert len(set(summaries)) == 3
    assert all(summary and "[UNK]" not in summary for summary in summaries)
    assert summaries[0] != "poor quality"
    assert any(phrase in summaries[2] for phrase in ("not worth", "waste", "broken", "terrible"))
