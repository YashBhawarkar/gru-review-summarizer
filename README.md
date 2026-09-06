# Threadline: trained GRU review summarizer

Threadline is a complete TensorFlow/Keras sequence-to-sequence project that turns
short English product reviews into title-like summaries. It uses a genuinely
trained bidirectional-GRU encoder-decoder with separate learned embeddings,
additive attention, teacher forcing, and stateful beam-search decoding. The
Streamlit app runs the included model
locally; it does not call an API, load a pretrained summarizer, train at startup,
or substitute extractive/canned text.

> **Scope:** this compact educational model is intended for short reviews in the
> clothing domain. It is not a long-document summarizer, and its output can be
> generic, incomplete, repetitive, or wrong.

## Current verification status

- **Implemented:** preprocessing, global deduplication and deterministic splits,
  train-only tokenizers, GRU training, checkpointing, model persistence,
  checksummed reload, length-normalized beam search, ROUGE evaluation, baseline,
  tests, Streamlit UI, and Colab workflow.
- **Trained:** TensorFlow 2.18.1 on 15,689 training examples. Early stopping
  restored epoch 5 after stopping at epoch 8.
- **Tested locally:** six automated tests pass; the saved model reloads and
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
4. filters to 4–110 review tokens and 1–10 title tokens;
5. shuffles once with a seeded NumPy generator, then creates 80/10/10
   train/validation/test splits; and
6. fits both Keras tokenizers **only on the training split**.

For the included seed-42 run, 23,486 raw rows yielded 19,675 complete pairs,
seven duplicate review bodies were removed, and 19,612 rows met the expanded
length contract. The final split is 15,689 / 1,961 / 1,962.

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
  --dataset-size 0 \
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
review IDs → Embedding(10,000 × 128, mask_zero) → BiGRU(192 × 2)
                                                    ├─→ projected encoder sequence
                                                    └─→ bridged decoder state

sostok/title IDs → Embedding(3,412 × 128, mask_zero)
                  → GRU(192, initial_state=bridged state)
                  → additive attention over the encoder sequence
                  → Dense(3,412 vocabulary logits)
```

The included model has **3,734,804 trainable parameters** (14.25 MiB of float32
parameters; the packaged `.keras` model is about 14.3 MiB). Teacher forcing shifts
the target sequence by one token. Inference feeds predicted decoder tokens back
one step at a time, passes each returned recurrent state forward, and uses a
two-candidate, length-normalized beam selected on validation data. It stops at
`eostok` or ten generated words and requires at least two words before accepting
the end marker.

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
  next autoregressive step;
- projects all bidirectional encoder outputs for additive attention and uses a
  validation-selected beam decoder instead of rebuilding inference layers; and
- persists one shared preprocessing configuration with artifact checksums.

## Honest held-out evaluation

The following are actual mean ROUGE F1 scores with stemming on all **1,962**
held-out examples. The test split was never used to fit tokenizers, train weights,
select a checkpoint, or select beam settings; those decisions used the validation
split. The previous shipped v1 model was re-scored on the exact same v2 test rows
for an apples-to-apples comparison.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Current bidirectional GRU + attention | **0.098357** | **0.023499** | **0.097710** |
| Previous unidirectional GRU (same rows) | 0.095177 | 0.020473 | 0.094536 |
| Lead first-sentence words (max 10) | 0.096878 | 0.018726 | 0.091429 |

The current model improves over v1 on all three measures (about +3.3% ROUGE-1,
+14.8% ROUGE-2, and +3.4% ROUGE-L) and narrowly beats the extractive baseline.
The absolute scores remain weak: even with attention, this small model often
learns high-frequency titles such as “love this dress” or confuses related product
types. The gain does not make it suitable for consequential use.

Representative held-out examples:

| Human title | GRU summary | Lead baseline |
|---|---|---|
| `great dress` | `great dress` | `comfortable great fit and beautiful colors the interesting thing is` |
| `great jeans for tall ladies` | `love these pants` | `these white jeans are super cute the longer length is` |
| `nice style but not on me` | `not for me` | `i received the vest and it was pretty much as` |

The source reviews and eight complete qualitative records are stored in
`artifacts/evaluation.json` and rendered in the technical section of the app.
ROUGE measures lexical overlap, not factuality, usefulness, or fluency.

## Artifact contract

The committed `artifacts/` directory contains:

- `model.keras` — uncompiled trained inference model (no optimizer state);
- `encoder_tokenizer.json` and `decoder_tokenizer.json`;
- `preprocessing.json` — exact sequence, vocabulary, architecture, and decoding
  settings;
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
This app keeps its packaged model modest, limits TensorFlow threads, caches one
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
