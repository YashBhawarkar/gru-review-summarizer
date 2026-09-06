"""Streamlit interface for the trained GRU review summarizer."""

from __future__ import annotations

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
    "Mixed headphone experience": (
        "I bought this wireless headphone set last week. The sound quality is amazing, with "
        "deep bass and clear highs. However, the battery life is terrible. It barely lasts 3 "
        "hours on a single charge. Also, the ear cups get very uncomfortable after about 30 "
        "minutes of wearing them. Customer service was helpful when I called, but overall I "
        "wouldn't recommend them for long trips."
    ),
    "Broken software update": (
        "This latest update completely broke the app for me. It crashes every time I try to "
        "open a new project. I've tried reinstalling, clearing the cache, and restarting my "
        "phone, but nothing works. The interface is also more confusing now and hides features "
        "that used to be one click away. I'm canceling until they fix these bugs."
    ),
    "Excellent kitchen scale": (
        "This compact kitchen scale is easy to use and gives the same reading every time. The "
        "display is bright, the buttons respond quickly, and the small size makes it easy to "
        "store. I have used it every day for a month and would gladly buy it again."
    ),
    "Mixed movie review": (
        "Visually stunning with breathtaking special effects and a fantastic soundtrack. The "
        "lead actor gave a stellar performance. Unfortunately, the plot was full of holes and "
        "the pacing was terribly slow in the middle act. It's worth watching on a big screen, "
        "but don't expect a deep storyline."
    ),
}
SERVICE_REVIEW_TERMS = {
    "ambiance",
    "breakfast",
    "dining",
    "hotel",
    "resort",
    "restaurant",
    "room",
    "stay",
    "waiter",
}


st.set_page_config(
    page_title="Threadline · Review Summarizer",
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
    if not path.exists():
        return PreprocessingConfig()
    try:
        return PreprocessingConfig.load(path)
    except (TypeError, ValueError) as exc:
        st.error(
            "The running application code and trained artifacts are from different revisions. "
            "The app owner should open Manage app and choose Reboot app."
        )
        st.caption(f"Configuration compatibility detail: {exc}")
        st.stop()


config = artifact_config()
st.markdown('<div class="hero-kicker">Review intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Turn a review into<br>a concise title.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Turn detailed customer feedback into a clear, title-like summary '
    'for review feeds, feedback dashboards, and faster product-quality triage.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="model-pill">One-click summaries</span>'
    f'<span class="model-pill">Up to {config.max_input_tokens} words</span>'
    '<span class="model-pill">No external API</span>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Threadline")
    st.caption("AI review title generator")
    artifacts_ready = (ARTIFACTS / "manifest.json").exists()
    if artifacts_ready:
        st.success("Ready")
    else:
        st.error("Trained artifacts missing")
    st.markdown("#### Designed for")
    st.write("Short English consumer-product reviews across electronics, home, media, and apparel.")
    st.markdown("#### Best results")
    st.write(
        f"Use one focused review of up to **{config.max_input_tokens} words** covering quality, "
        "usefulness, defects, value, or the reason for a return."
    )


st.markdown("### Write or choose a review")
if "review_text" not in st.session_state:
    st.session_state.review_text = SAMPLES["Mixed headphone experience"]


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
    max_chars=3_000,
    placeholder="Describe the product, its quality, and your overall experience…",
)

clean_word_count = len(normalize_text(review).split())
normalized_words = set(normalize_text(review).split())
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
if len(normalized_words & SERVICE_REVIEW_TERMS) >= 2:
    st.warning(
        "This looks like a hospitality or service review. That domain was not represented in "
        "training, so the model may produce a generic title or misidentify the subject."
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
                    st.caption("AI-GENERATED SUMMARY")
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
st.markdown("### How it works in practice")
step_one, step_two, step_three = st.columns(3)
with step_one:
    with st.container(border=True):
        st.markdown("**1 · Collect feedback**")
        st.caption("A customer submits a short product review after a purchase.")
with step_two:
    with st.container(border=True):
        st.markdown("**2 · Generate a title**")
        st.caption("Threadline converts the review into a concise, scannable headline.")
with step_three:
    with st.container(border=True):
        st.markdown("**3 · Put it to work**")
        st.caption("Show it on review cards or use it in feedback and moderation queues.")

st.caption(
    "In a production workflow, the same summarizer can run when a review is submitted, then "
    "store the generated title alongside the original review for optional human editing."
)
