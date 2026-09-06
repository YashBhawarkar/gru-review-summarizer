# Threadline: trained GRU review summarizer

Threadline is a complete TensorFlow/Keras sequence-to-sequence project that turns
short English product reviews into title-like summaries. It uses a genuinely
trained GRU encoder-decoder with separate learned embeddings, teacher forcing,
and stateful autoregressive decoding. The Streamlit app runs the included model
locally; it does not call an API, load a pretrained summarizer, train at startup,
or substitute extractive/canned text.

> **Scope:** this compact educational model is intended for short reviews in the
> clothing domain. It is not a long-document summarizer, and its output can be
> generic, incomplete, repetitive, or wrong.

## Current verification status

- **Implemented:** preprocessing, global deduplication and deterministic splits,
  train-only tokenizers, GRU training, checkpointing, model persistence,
  checksummed reload, greedy decoding, ROUGE evaluation, baseline, tests,
  Streamlit UI, and Colab workflow.
- **Trained:** TensorFlow 2.18.1 on 10,334 training examples. Early stopping
  restored epoch 10 after stopping at epoch 13.
- **Tested locally:** five automated tests pass; the saved model reloads and
  generates a real summary. The Streamlit app is also run and browser-checked as
  part of this repository's handoff.
- **Public deployment:** requires the owner's GitHub and Streamlit Community
  Cloud account connection. Do not infer a deployment from this repository; a
  URL should be reported only after it has been opened and verified.

## Try the included trained model

Python 3.12 is recommended (3.10–3.12 are supported by the project).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, select a sample or enter a short review,
and click **Generate summary**. Model loading is cached with
`st.cache_resource`. The app verifies SHA-256 checksums before loading the model.

Run the checks with the training dependencies installed:

```bash
pip install -r requirements-train.txt
pytest -q
```

## Dataset, availability, and license

Training uses [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews),
a 23,486-row review dataset whose `Review Text` is the source and human-written
`Title` is the summary target. The dataset is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain).
The Kaggle UI may ask for an account, so the downloader uses a public,
credential-free, CC0-tagged Parquet conversion on Hugging Face:

- Repository: `chibifire/kaggle-womens-ecom-clothing-reviews`
- Pinned revision: `81458f32611e8f8d78539b5d44cd4c2dc2c98000`
- Parquet SHA-256: `2350fc698612b568149115425014a94565ea46daa5d3e6136989876d8f4a1637`

Download and verify it reproducibly:

```bash
python -m scripts.download_data
```

The raw dataset is intentionally gitignored. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for complete attribution.

## Data preparation and leakage controls

The training script:

1. removes missing/empty review-title pairs;
2. applies the same Unicode, case, punctuation, apostrophe, and whitespace
   normalization used by inference;
3. deduplicates the entire eligible pool by normalized review text **before**
   any sampling or split;
4. filters to 4–80 review tokens and 1–8 title tokens;
5. shuffles once with a seeded NumPy generator, then creates 80/10/10
   train/validation/test splits; and
6. fits both Keras tokenizers **only on the training split**.

For the included seed-42 run, 23,486 raw rows yielded 19,675 complete pairs,
seven duplicate review bodies were removed, and 12,918 rows met the length
contract. The final split is 10,334 / 1,291 / 1,293.

Padding is token ID 0 and is masked in embeddings, loss weighting, and token
accuracy. `<unk>` is a real OOV token. `sostok` and `eostok` are included before
target tokenization, verified after reload, supplied to the decoder correctly,
and hidden from displayed summaries.

## Train from scratch

Training is separate from the hosted application.

```bash
source .venv/bin/activate
pip install -r requirements-train.txt
python -m scripts.download_data
python -m scripts.train \
  --dataset-size 13000 \
  --epochs 20 \
  --batch-size 64 \
  --seed 42
python -m scripts.evaluate
```

All requested controls are CLI flags; run `python -m scripts.train --help` for
the full set, including vocabulary sizes, embedding/hidden dimensions, learning
rate, patience, and sequence limits. `--dataset-size 0` uses all eligible rows.

Training uses:

- best-validation-loss weight checkpoints;
- early stopping with best-weight restoration;
- `ReduceLROnPlateau`;
- per-timestep padding weights;
- deterministic Python, NumPy, and TensorFlow seeds where the platform supports
  deterministic kernels; and
- a CSV log plus JSON training history.

For free GPU training, open [notebooks/train_in_colab.ipynb](notebooks/train_in_colab.ipynb)
in Google Colab. It installs the pinned environment, downloads the verified CC0
data, trains, evaluates, verifies reload/inference, and downloads a ZIP of the
artifacts. To import that ZIP into a checkout, extract its files directly into
`artifacts/`, then run `pytest -q` and `streamlit run app.py`.

## Model architecture

```text
review IDs → Embedding(8,000 × 96, mask_zero) → GRU(160) → encoder state

sostok/title IDs → Embedding(2,490 × 96, mask_zero)
                  → GRU(160, initial_state=encoder state)
                  → Dense(2,490 vocabulary logits)
```

The included model has **1,655,610 trainable parameters** (6.32 MiB of float32
parameters; the packaged `.keras` model is about 6.4 MiB). Teacher forcing shifts
the target sequence by one token. Inference feeds one predicted decoder token at
a time, passes the returned recurrent state into the next step, and stops at
`eostok` or eight generated words.

### Corrections to the reference notebook

The Packt notebook demonstrates the central seq2seq idea but refits the encoder
tokenizer on test text, has brittle special-token/filter behavior, mixes names
from a translation example, and manually reconstructs inference in a way that is
easy to miswire. This project:

- never calls `fit_on_texts` outside the training-only tokenizer function;
- persists and reloads both tokenizers instead of mutating them at inference;
- sizes the output layer from the decoder vocabulary;
- uses explicit padding and OOV IDs and masks padding during optimization;
- preserves and validates start/end IDs across serialization;
- passes encoder state into the decoder and each returned decoder state into the
  next autoregressive step; and
- persists one shared preprocessing configuration with artifact checksums.

## Honest held-out evaluation

The following are actual mean ROUGE F1 scores with stemming on all **1,293**
held-out examples. The test split was never used to fit tokenizers, train weights,
select a checkpoint, or tune early stopping.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Trained GRU encoder-decoder | 0.079385 | 0.012841 | 0.079385 |
| Lead first-sentence words (max 8) | **0.110566** | **0.022218** | **0.104690** |

The simple extractive baseline is stronger. This is a meaningful negative result:
a small, no-attention GRU trained on about ten thousand examples tends to learn
high-frequency titles such as “beautiful dress” and “great fit,” while the lead
baseline often copies a title word directly from the review.

Representative held-out examples:

| Human title | GRU summary | Lead baseline |
|---|---|---|
| `great top` | `great top` | `one of the best tops i have ever` |
| `lovely top but too tight` | `beautiful dress` | `the top is a pretty design it's a` |
| `did not look good on me` | `beautiful dress` | `this is a beautiful dress the quality is` |

The source reviews and six complete qualitative records are stored in
`artifacts/evaluation.json` and rendered in the technical section of the app.
ROUGE measures lexical overlap, not factuality, usefulness, or fluency.

## Artifact contract

The committed `artifacts/` directory contains:

- `model.keras` — uncompiled trained inference model (no optimizer state);
- `encoder_tokenizer.json` and `decoder_tokenizer.json`;
- `preprocessing.json` — exact sequence, vocabulary, and architecture settings;
- `training_history.json` and `training_log.csv`;
- `evaluation.json` — real metrics and examples; and
- `manifest.json` — metadata and SHA-256 for every runtime/evaluation artifact.

If these files are absent or altered, the app shows a setup/integrity error. It
does not create random weights or return a baseline in place of the GRU.

## Free Streamlit Community Cloud deployment

Deployment guidance was verified against the official Streamlit documentation on
2026-09-06. Community Cloud is free, deploys from GitHub, runs the selected
Python entrypoint from the repository root, and defaults to Python 3.12. The
platform currently documents approximate shared limits of 0.078–2 CPU cores,
690 MB–2.7 GB RAM, and up to 50 GB storage; Streamlit warns these can change.
This app keeps its packaged model small, limits TensorFlow threads, caches one
model instance, and never loads training dependencies or data.

Official references:

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Status and limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)
- [Resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

To deploy after pushing this complete directory to a **public GitHub repository**:

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Click **Create app**, then choose the repository and branch.
3. Set the entrypoint to `app.py`.
4. In **Advanced settings**, select Python **3.12**. No secrets are needed.
5. Deploy, wait for `artifacts/model.keras` to load, generate a summary, and open
   the public `https://…streamlit.app` URL in a fresh browser session.

Only report that URL after that final request and inference both succeed. If the
model is moved to external storage later, pin an immutable URL and update the
manifest checksum rather than downloading an unverified “latest” file.

## Repository layout

```text
app.py                         Streamlit app (inference only)
gru_summarizer/                preprocessing, model, persistence, inference
scripts/download_data.py       pinned and checksummed CC0 download
scripts/train.py               configurable teacher-forced training
scripts/evaluate.py            held-out ROUGE + extractive baseline
notebooks/train_in_colab.ipynb free Colab training/export path
tests/                         preprocessing, split, reload/inference checks
artifacts/                     trained model, tokenizers, metrics, manifest
requirements*.txt              pinned app, training, and Colab environments
```

## Attribution

The implementation is inspired by Section 5 of Packt Publishing's
[Advanced NLP Projects with TensorFlow 2.0](https://github.com/PacktPublishing/Advanced-NLP-Projects-with-TensorFlow-2.0),
which is MIT-licensed (Copyright © 2018 Packt). This project retains that notice
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and is itself MIT-licensed.
