"""Streamlit interface for the trained GRU review-title generator."""

from __future__ import annotations

import hashlib
from html import escape
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
BOOTSTRAP_CSS = ROOT / "assets" / "bootstrap-5.3.8.min.css"
APP_CSS = ROOT / "assets" / "threadline.css"
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
    page_title="Threadline · Review Title Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for stylesheet in (BOOTSTRAP_CSS, APP_CSS):
    if stylesheet.exists():
        st.html(stylesheet)


@st.cache_resource(show_spinner=False)
def cached_summarizer(artifact_path: str, manifest_fingerprint: str):
    """Load assets once per manifest version and checksum-verify every reload."""
    del manifest_fingerprint  # The value is intentionally part of Streamlit's cache key.
    return load_summarizer(artifact_path, verify_checksums=True)


def artifact_fingerprint() -> str:
    """Return a stable cache-busting key for the currently deployed artifacts."""
    manifest_path = ARTIFACTS / "manifest.json"
    if not manifest_path.exists():
        return "missing"
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


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
artifacts_ready = (ARTIFACTS / "manifest.json").exists()
status_class = "" if artifacts_ready else " is-missing"
status_copy = "GRU model ready" if artifacts_ready else "Artifacts missing"

st.html(
    f"""
    <header aria-label="Threadline introduction">
      <div class="tl-navbar" aria-label="Application status">
        <div class="tl-brand">
          <span class="tl-brand-mark" aria-hidden="true">✦</span>
          <span>Threadline</span>
        </div>
        <span class="tl-status{status_class}" role="status">
          <span class="tl-status-dot" aria-hidden="true"></span>
          {status_copy}
        </span>
      </div>

      <section class="tl-hero">
        <div class="tl-hero-content">
          <div class="row align-items-center g-4">
            <div class="col-lg-8">
              <div class="tl-kicker">Review intelligence · GRU + attention</div>
              <h1>Turn a review into a concise title.</h1>
              <p class="tl-hero-copy">
                Transform one short consumer-product review into a compact, customer-style
                headline for review cards, feedback dashboards, and product-quality queues.
              </p>
              <div class="d-flex flex-wrap gap-2 mt-4" aria-label="Model capabilities">
                <span class="badge rounded-pill tl-feature-badge">One-click titles</span>
                <span class="badge rounded-pill tl-feature-badge">Up to {config.max_input_tokens} words</span>
                <span class="badge rounded-pill tl-feature-badge">2–10 word output</span>
                <span class="badge rounded-pill tl-feature-badge">No external API</span>
              </div>
            </div>
            <div class="col-lg-4">
              <article class="card tl-preview-card" aria-label="Example output shape">
                <span class="tl-preview-label">Example output shape</span>
                <p class="tl-preview-review">
                  “The sound is excellent, though the short battery life is disappointing.”
                </p>
                <div class="tl-preview-title">good product poor quality</div>
              </article>
            </div>
          </div>
        </div>
      </section>

      <div class="alert tl-contract-alert" role="note">
        <span class="tl-contract-icon" aria-hidden="true">i</span>
        <span><strong>Title-generation contract:</strong> Threadline creates one short,
        Amazon-style headline. It does not attempt to restate every positive and negative
        detail from the review.</span>
      </div>
    </header>
    """
)


with st.sidebar:
    st.html(
        """
        <section class="tl-sidebar-card">
          <span class="tl-brand-mark" aria-hidden="true">✦</span>
          <h2>Threadline</h2>
          <p>A compact review-title generator powered by a trained GRU encoder–decoder.</p>
        </section>
        """
    )
    if artifacts_ready:
        st.success("Trained model ready")
    else:
        st.error("Trained artifacts missing")
    st.html(
        f"""
        <div class="tl-sidebar-list">
          <h3 class="h6 fw-bold mt-4">Model contract</h3>
          <ul class="list-group list-group-flush">
            <li class="list-group-item">Short English consumer-product reviews</li>
            <li class="list-group-item">First {config.max_input_tokens} normalized words</li>
            <li class="list-group-item">One title of roughly 2–10 words</li>
            <li class="list-group-item">Electronics, home, media, and apparel</li>
          </ul>
        </div>
        """
    )
    st.info(
        "Best results come from one focused review covering quality, usefulness, defects, "
        "value, or the reason for a return."
    )


st.html(
    """
    <div class="tl-section-heading">
      <div>
        <span class="badge rounded-pill tl-step-badge">01 · Review</span>
        <h2>Write or choose a review</h2>
      </div>
      <p>Keep it focused for the clearest title.</p>
    </div>
    """
)
if "review_text" not in st.session_state:
    st.session_state.review_text = SAMPLES["Mixed headphone experience"]


def choose_sample() -> None:
    st.session_state.review_text = SAMPLES[st.session_state.sample_choice]
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_review", None)


with st.container(key="review_workspace"):
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

    normalized_review = normalize_text(review)
    clean_word_count = len(normalized_review.split())
    normalized_words = set(normalized_review.split())
    word_meter = min(100, round(100 * clean_word_count / config.max_input_tokens))
    meter_class = " is-over" if clean_word_count > config.max_input_tokens else ""
    st.html(
        f"""
        <div class="tl-word-meter{meter_class}">
          <div class="tl-word-meter-copy">
            <span>Only the first {config.max_input_tokens} normalized words are used</span>
            <strong>{clean_word_count} / {config.max_input_tokens} words</strong>
          </div>
          <div class="progress" role="progressbar" aria-label="Input word usage"
               aria-valuenow="{min(clean_word_count, config.max_input_tokens)}"
               aria-valuemin="0" aria-valuemax="{config.max_input_tokens}">
            <div class="progress-bar" style="width: {word_meter}%"></div>
          </div>
        </div>
        """
    )

    if clean_word_count > config.max_input_tokens:
        st.warning(
            f"This review will be shortened to its first {config.max_input_tokens} normalized "
            "words before inference."
        )
    if len(normalized_words & SERVICE_REVIEW_TERMS) >= 2:
        st.warning(
            "This looks like a hospitality or service review. That domain was not represented "
            "in training, so the model may produce a generic title or misidentify the subject."
        )

    generate = st.button("Generate title", type="primary", width="stretch")
    if generate:
        if not review.strip():
            st.error("Add a review before generating a title.")
        elif not artifacts_ready:
            st.error(
                "No trained model is installed. Run `python -m scripts.download_data`, then "
                "`python -m scripts.train` and `python -m scripts.evaluate`. The app will never "
                "substitute random weights or a canned fallback."
            )
        else:
            try:
                with st.spinner("Reading the review and decoding a title…"):
                    result = cached_summarizer(
                        str(ARTIFACTS), artifact_fingerprint()
                    ).summarize(review)
                st.session_state.last_result = result
                st.session_state.last_review = review
            except Exception as exc:
                st.error(f"The trained artifacts could not produce a title: {exc}")


result = st.session_state.get("last_result")
result_review = st.session_state.get("last_review")
if result is not None and result_review == review:
    display_title = result.summary or "No tokens generated before the end marker."
    safe_review = escape(result_review)
    safe_title = escape(display_title)
    title_words = len(result.summary.split())
    reduction = 100 * (1 - title_words / max(result.input_tokens, 1))

    st.html(
        """
        <div class="tl-section-heading">
          <div>
            <span class="badge rounded-pill tl-step-badge">02 · Result</span>
            <h2>Your generated review title</h2>
          </div>
          <p>Review the headline before publishing it.</p>
        </div>
        """
    )
    st.html(
        f"""
        <section class="row g-3 tl-result-grid" aria-label="Generated result"
                 aria-live="polite">
          <div class="col-lg-7">
            <article class="card tl-original-card">
              <div class="card-body">
                <span class="tl-card-label">Original review</span>
                <p class="tl-review-copy">{safe_review}</p>
              </div>
            </article>
          </div>
          <div class="col-lg-5">
            <article class="card tl-generated-card">
              <div class="card-body d-flex flex-column justify-content-between">
                <div>
                  <span class="tl-card-label">AI-generated review title</span>
                  <div class="tl-generated-title">{safe_title}</div>
                </div>
                <div><span class="badge rounded-pill tl-model-badge">Trained GRU output</span></div>
              </div>
            </article>
          </div>
        </section>
        <section class="row g-3 mt-1" aria-label="Inference measurements">
          <div class="col-md-4">
            <div class="card tl-stat-card">
              <div class="tl-stat-label">Input words</div>
              <div class="tl-stat-value">{result.input_tokens}</div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card tl-stat-card">
              <div class="tl-stat-label">Title words</div>
              <div class="tl-stat-value">{title_words}</div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card tl-stat-card">
              <div class="tl-stat-label">Length reduction</div>
              <div class="tl-stat-value">{reduction:.0f}%</div>
            </div>
          </div>
        </section>
        <p class="tl-inference-note">Measured model inference time:
          {result.inference_seconds * 1000:.1f} ms</p>
        """
    )
    if not result.summary:
        st.caption("The empty output is shown honestly; no fallback text was substituted.")
    if result.truncated:
        st.info(f"Only the first {result.used_tokens} normalized words reached the encoder.")


st.html(
    """
    <div class="tl-section-heading">
      <div>
        <span class="badge rounded-pill tl-step-badge">Workflow</span>
        <h2>How it works in practice</h2>
      </div>
      <p>From raw feedback to a scannable headline.</p>
    </div>
    <section class="row g-3" aria-label="Application workflow">
      <div class="col-md-4">
        <article class="card tl-step-card">
          <span class="tl-step-number" aria-hidden="true">1</span>
          <h3>Collect feedback</h3>
          <p>A customer submits one focused product review after a purchase.</p>
        </article>
      </div>
      <div class="col-md-4">
        <article class="card tl-step-card">
          <span class="tl-step-number" aria-hidden="true">2</span>
          <h3>Generate a title</h3>
          <p>The trained GRU decodes a concise, customer-style headline.</p>
        </article>
      </div>
      <div class="col-md-4">
        <article class="card tl-step-card">
          <span class="tl-step-number" aria-hidden="true">3</span>
          <h3>Review and publish</h3>
          <p>Store the title with the original review, with optional human editing.</p>
        </article>
      </div>
    </section>
    <footer class="tl-footer">
      Threadline · TensorFlow GRU encoder–decoder · Runs locally with no external inference API
    </footer>
    """
)
