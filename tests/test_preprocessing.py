import json

import pytest

from gru_summarizer.config import PreprocessingConfig
from gru_summarizer.preprocessing import (
    add_boundaries,
    encode_sources,
    make_tokenizers,
    normalize_text,
    validate_special_tokens,
)


def test_normalization_is_deterministic_and_keeps_apostrophes():
    assert normalize_text("  It’s <b>GREAT</b> & comfy! ") == "it's b great b comfy"


def test_boundaries_and_oov_survive_round_trip():
    config = PreprocessingConfig(max_input_tokens=5, max_summary_tokens=6)
    enc, dec = make_tokenizers(
        ["soft blue dress", "poor zipper quality"],
        ["lovely dress", "bad zipper"],
        config,
    )
    validate_special_tokens(dec, config)
    bounded = add_boundaries("Lovely dress", config)
    ids = dec.texts_to_sequences([bounded])[0]
    assert ids[0] == dec.word_index[config.start_token]
    assert ids[-1] == dec.word_index[config.end_token]
    unknown_id = enc.texts_to_sequences(["never_seen_before"])[0][0]
    assert unknown_id == enc.word_index[config.oov_token]


def test_source_encoding_post_pads_and_post_truncates():
    config = PreprocessingConfig(max_input_tokens=3)
    enc, _ = make_tokenizers(["one two three four"], ["short"], config)
    encoded = encode_sources(["one two three four", "one"], enc, config)
    assert encoded.shape == (2, 3)
    assert encoded[0].tolist() == enc.texts_to_sequences(["one two three"])[0]
    assert encoded[1, 1:].tolist() == [0, 0]


def test_future_artifact_schema_reports_a_clear_compatibility_error(tmp_path):
    payload = PreprocessingConfig().to_dict()
    payload["future_architecture_option"] = True
    path = tmp_path / "preprocessing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires newer application code"):
        PreprocessingConfig.load(path)
