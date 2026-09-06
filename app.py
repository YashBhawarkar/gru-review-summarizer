"""Streamlit interface for the trained GRU review summarizer."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

import streamlit as st

from gru_summarizer.config import PreprocessingConfig
from gru_summarizer.inference import load_summarizer
from gru_summarizer.preprocessing import normalize_text


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SAMPLES = {
    "Soft, flattering dress": (
        "I was pleasantly surprised by this dress. The fabric is soft without feeling thin, "
        "the waist is flattering, and the color matches the photos. It fit true to size and "
        "stayed comfortable through an entire evening."
    ),
    "Disappointing zipper": (
        "The jacket looked beautiful when it arrived, but the zipper caught on the lining "
        "every time I used it and broke after two wears. The sleeves also run a little short. "
        "I wanted to love it, but the construction is disappointing."
    ),
    "Runs small": (
        "This top has a lovely print and the material feels nice, but it runs at least one size "
        "small through the shoulders. I normally wear a medium and could barely move my arms. "
        "Order up if you want a relaxed fit."
    ),
}


st.set_page_config(
    page_title="Threadline · GRU Review Summarizer",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f4f2ff 0px, #f7f7fb 280px); }
      .block-container { max-width: 940px; padding-top: 2.1rem; padding-bottom: 3rem; }
      .hero-kicker { color: #6d5ce7; font-size: .78rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
      .hero-title { color: #1e2032; font-size: clamp(2.3rem, 6vw, 4.2rem); line-height: .98; letter-spacing: -.055em; font-weight: 760; margin: .55rem 0 .9rem; }
      .hero-copy { color: #5a5f72; font-size: 1.05rem; line-height: 1.65; max-width: 720px; margin-bottom: 1.8rem; }
      .model-pill { display: inline-block; background: #ebe7ff; color: #5947d7; border: 1px solid #d9d1ff; border-radius: 999px; padding: .35rem .7rem; font-size: .76rem; font-weight: 700; margin-right: .35rem; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.85); border-color: #e5e2f1; box-shadow: 0 10px 30px rgba(52,43,105,.06); border-radius: 16px; }
      div[data-testid="stMetric"] { background: #fff; border: 1px solid #ebe9f2; padding: .9rem; border-radius: 12px; }
      div.stButton > button[kind="primary"] { background: #6d5ce7; border: none; min-height: 3rem; font-weight: 700; }
      .summary-box { background: #27233e; color: #fff; border-radius: 14px; padding: 1.3rem 1.4rem; font-size: 1.22rem; line-height: 1.5; min-height: 84px; }
      .small-note { color: #737789; font-size: .82rem; line-height: 1.5; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_summarizer(artifact_path: str):
    """Load and checksum-verify model assets once per Streamlit process."""
    return load_summarizer(artifact_path, verify_checksums=True)


def artifact_config() -> PreprocessingConfig:
    path = ARTIFACTS / "preprocessing.json"
    return PreprocessingConfig.load(path) if path.exists() else PreprocessingConfig()


def evaluation_payload() -> dict:
    path = ARTIFACTS / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


config = artifact_config()
st.markdown('<div class="hero-kicker">TensorFlow · Sequence to sequence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Turn a review into<br>a concise title.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">A compact, locally running GRU encoder–decoder trained on '
    'human-written clothing review titles. No API calls, no pretrained language model, and '
    'no training during app startup.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="model-pill">GRU encoder + decoder</span>'
    '<span class="model-pill">Greedy autoregressive decoding</span>'
    '<span class="model-pill">CPU friendly</span>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Threadline")
    st.caption("Short-review abstractive summarization")
    artifacts_ready = (ARTIFACTS / "manifest.json").exists()
    if artifacts_ready:
        st.success("Trained artifacts available")
    else:
        st.error("Trained artifacts missing")
    st.markdown("#### Designed for")
    st.write("Short English product reviews, especially apparel and fit feedback.")
    st.markdown("#### Input contract")
    st.write(f"The model uses the first **{config.max_input_tokens} normalized words**.")
    st.markdown("#### Important")
    st.caption(
        "This small educational model can be generic, repetitive, or inaccurate. "
        "It is not intended for long documents or consequential decisions."
    )


st.markdown("### Write or choose a review")
if "review_text" not in st.session_state:
    st.session_state.review_text = SAMPLES["Soft, flattering dress"]


def choose_sample() -> None:
    st.session_state.review_text = SAMPLES[st.session_state.sample_choice]


st.selectbox(
    "Sample review",
    list(SAMPLES),
    key="sample_choice",
    on_change=choose_sample,
    help="Samples are original examples written for this demo, not memorized training rows.",
)
review = st.text_area(
    "Review text",
    key="review_text",
    height=190,
    max_chars=2_500,
    placeholder="Describe the product, fit, quality, and your overall impression…",
)

clean_word_count = len(normalize_text(review).split())
notice_col, count_col = st.columns([3, 1])
with notice_col:
    st.caption(
        f"The model was trained for short reviews. Text after word {config.max_input_tokens} is not used."
    )
with count_col:
    st.caption(f"{clean_word_count} normalized words")
if clean_word_count > config.max_input_tokens:
    st.warning(
        f"This review will be shortened to its first {config.max_input_tokens} normalized words before inference."
    )

generate = st.button("Generate summary", type="primary", width="stretch")
if generate:
    if not review.strip():
        st.error("Add a review before generating a summary.")
    elif not artifacts_ready:
        st.error(
            "No trained model is installed. Run `python -m scripts.download_data`, then "
            "`python -m scripts.train` and `python -m scripts.evaluate`. The app will never "
            "substitute random weights or a canned fallback."
        )
    else:
        try:
            with st.spinner("Reading the review and decoding a title…"):
                result = cached_summarizer(str(ARTIFACTS)).summarize(review)
            st.markdown("### Result")
            with st.container(border=True):
                left, right = st.columns([1, 1])
                with left:
                    st.caption("ORIGINAL REVIEW")
                    st.write(review)
                with right:
                    st.caption("GRU-GENERATED SUMMARY")
                    display_summary = result.summary or "No tokens generated before the end marker."
                    st.markdown(
                        f'<div class="summary-box">{display_summary}</div>', unsafe_allow_html=True
                    )
                    if not result.summary:
                        st.caption("The empty output is shown honestly; no fallback text was substituted.")

            summary_words = len(result.summary.split())
            reduction = 100 * (1 - summary_words / max(result.input_tokens, 1))
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Input words", result.input_tokens)
            metric2.metric("Summary words", summary_words)
            metric3.metric("Length reduction", f"{reduction:.0f}%")
            st.caption(f"Measured model inference time: {result.inference_seconds * 1000:.1f} ms")
            if result.truncated:
                st.info(f"Only the first {result.used_tokens} normalized words were passed to the encoder.")
        except Exception as exc:
            st.error(f"The trained artifacts could not produce a summary: {exc}")


st.divider()
st.markdown("### About this model")
st.write(
    "The encoder embeds up to 80 normalized words and compresses them with a GRU. During "
    "training, the decoder receives the previous human title token (teacher forcing). During "
    "inference, it starts from `sostok`, predicts one token at a time, carries the recurrent "
    "state forward, and stops at `eostok` or the learned length limit."
)

with st.expander("Architecture and genuine evaluation", expanded=False):
    st.code(
        "Review tokens → Embedding(96) → GRU(160) → hidden state\n"
        "sostok → Embedding(96) → GRU(160, encoder state) → vocabulary logits\n"
        "                                      ↳ predicted token + next state",
        language="text",
    )
    evaluation = evaluation_payload()
    if evaluation:
        st.markdown(f"**Held-out examples evaluated:** {evaluation['examples_evaluated']:,}")
        st.caption(evaluation["metric"])
        rows = []
        for method_key, label in (("gru", "GRU encoder–decoder"), ("lead_words_baseline", "Lead-words baseline")):
            metrics = evaluation[method_key]
            rows.append(
                {
                    "Method": label,
                    "ROUGE-1": metrics["rouge1"],
                    "ROUGE-2": metrics["rouge2"],
                    "ROUGE-L": metrics["rougeL"],
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        st.markdown("**Representative test examples**")
        for example in evaluation.get("qualitative_examples", [])[:3]:
            st.caption("REVIEW")
            st.write(example["review"])
            st.write(f"**Human title:** {example['reference_title']}")
            st.write(f"**GRU:** {example['gru_summary'] or '∅ (empty)'}")
            st.write(f"**Baseline:** {example['lead_baseline']}")
            st.divider()
    else:
        st.info("Evaluation results will appear here after the held-out evaluation script runs.")

st.markdown(
    '<div class="small-note">Dataset: Women’s E-Commerce Clothing Reviews (CC0). '
    'Implementation inspired by Packt Publishing’s MIT-licensed Advanced NLP Projects with '
    'TensorFlow 2.0, Section 5. See the repository README and third-party notices for full attribution.</div>',
    unsafe_allow_html=True,
)
