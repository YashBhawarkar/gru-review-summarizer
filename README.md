# Threadline: trained GRU review summarizer

Threadline is a complete TensorFlow/Keras sequence-to-sequence project that turns
short English product reviews into title-like summaries. It uses a genuinely
trained bidirectional-GRU encoder-decoder with separate learned embeddings,
additive attention, teacher forcing, and stateful autoregressive decoding. The
Streamlit app runs the included model in-process; it does not call an API, load
a pretrained summarizer, train at startup, or substitute extractive/canned text.

**Live demo:** [gru-review-summarizer.streamlit.app](https://gru-review-summarizer.streamlit.app)

> **Scope:** this compact educational model is intended for short English
> consumer-product reviews. It is not a restaurant/service or long-document
> summarizer, and its output can be generic, incomplete, repetitive, or wrong.

## Current verification status

- **Implemented:** preprocessing, global deduplication and deterministic splits,
  train-only tokenizers, GRU training, checkpointing, model persistence,
  checksummed reload, configurable beam search, ROUGE evaluation, baseline,
  tests, Streamlit UI, and Colab workflow.
- **Trained:** TensorFlow 2.18.1 on 94,865 training examples spanning varied
  consumer products. Early stopping restored epoch 4 after stopping at epoch 6.
- **Tested locally:** all nine automated checks pass; the saved model reloads and
  generates a real summary. The Streamlit app is also run and browser-checked as
  part of this repository's handoff.
- **Public deployment:** the Community Cloud app is live and was browser-verified
  after a clean model-v3 rebuild. A real model inference completed successfully
  with no browser-console errors.

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

Training uses the [Amazon Reviews Polarity distribution](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews),
which contains 3.6 million training rows with a human-written review title and
body. The Kaggle distribution is published under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) and is pinned to
dataset version 2. It was assembled by Xiang Zhang, Junbo Zhao, and Yann LeCun
from Amazon review data introduced by Julian McAuley and Jure Leskovec.

The credential-free downloader scans the source CSV in 100,000-row chunks and
creates a fixed 120,000-row sample. The default sampled Parquet has SHA-256
`4743846b93a68e9295aa1dd3cb1dd982cb7861a838aa7cbf7be7ffd97630ebc0`.
The full source archive is about 1.29 GiB, so allow enough local/Colab storage.

Download and verify it reproducibly:

```bash
python -m scripts.download_data
```

The source archive, download cache, sampled rows, and processed splits are all
gitignored. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for complete
attribution and source links.

## Data preparation and leakage controls

The training script:

1. removes missing/empty review-title pairs;
2. applies the same Unicode, case, punctuation, apostrophe, and whitespace
   normalization used by inference;
3. deduplicates the sampled pool by normalized review text **before** any
   train/validation/test split;
4. filters to 4–120 review tokens and 1–10 title tokens;
5. shuffles once with a seeded NumPy generator, then creates 80/10/10
   train/validation/test splits; and
6. fits both Keras tokenizers **only on the training split**.

For the included seed-42 run, the sampler scanned all 3.6 million source training
rows and wrote 120,000 candidate pairs. Normalization left 118,582 eligible
pairs with 90,734 distinct titles. The final split is 94,865 / 11,858 / 11,859.

Padding is token ID 0 and is masked in embeddings, loss weighting, and token
accuracy. `<unk>` is a real input/target OOV token and is masked from decoder
selection so it is never displayed as a generated word. `sostok` and `eostok`
are included before target tokenization, verified after reload, supplied to the
decoder correctly, and hidden from displayed summaries.

## Train from scratch

Training is separate from the hosted application.

```bash
source .venv/bin/activate
pip install -r requirements-train.txt
python -m scripts.download_data
python -m scripts.train \
  --dataset-size 0 \
  --epochs 10 \
  --batch-size 128 \
  --seed 42 \
  --patience 2
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
review IDs → Embedding(20,000 × 128, mask_zero) → BiGRU(192 × 2)
                                                    ├─→ projected encoder sequence
                                                    └─→ bridged decoder state

sostok/title IDs → Embedding(6,000 × 128, mask_zero)
                  → GRU(192, initial_state=bridged state)
                  → additive attention over the encoder sequence
                  → Dense(6,000 vocabulary logits)
```

The included model has **6,342,448 trainable parameters** (24.19 MiB of float32
parameters; the packaged `.keras` model is about 24 MiB). Teacher forcing shifts
the target sequence by one token. Inference feeds predicted decoder tokens back
one step at a time, passes each returned recurrent state forward, and uses a
single-path decoder selected on the validation data. It still carries each GRU
state forward autoregressively and enforces a two-word minimum; configurable beam
search remains implemented for experiments. Decoding stops at `eostok` or ten
generated words.

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
  validation-checked autoregressive decoder instead of rebuilding inference layers; and
- persists one shared preprocessing configuration with artifact checksums.

## Honest held-out evaluation

The following are actual mean ROUGE F1 scores with stemming on all **11,859**
held-out broad-product examples. The test split was never used to fit tokenizers,
train weights, or select a checkpoint. Decoder alternatives were compared on the
11,858-row validation split. The previous apparel model was re-scored on the same
broad test rows for an apples-to-apples domain comparison.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Current broad-product BiGRU + attention | **0.082088** | **0.014812** | **0.081313** |
| Previous apparel BiGRU (same broad rows) | 0.027739 | 0.003517 | 0.027447 |
| Lead words baseline (max 10) | 0.101038 | 0.027788 | 0.095344 |

The new model substantially improves over the apparel model on broad reviews,
but it **does not beat the simple extractive baseline**. Absolute scores remain
weak: human review titles are highly subjective, and the compact GRU still learns
frequent phrases such as “great product” and “not worth the money.” This is an
honest educational result, not a production-quality claim.

Representative held-out examples:

| Human title | GRU summary | Lead baseline |
|---|---|---|
| `not for exploratory kids` | `not worth the money` | `i bought the little touch when my daughter was 14` |
| `great pillows` | `great product` | `as a huge d backs fan i decided i needed` |
| `this book is absolutely lousy` | `worst book ever` | `this is one of the worst books i have ever` |

The product-domain stress tests now decode as `good product poor quality` for
the mixed headphone review, `a good read` for the fantasy book, `a great film`
for the movie, and `don't waste your money` for the broken app. The service-domain
tests remain unreliable: the hotel becomes `great for the price`, while the
restaurant is incorrectly labeled `a great book`. The app therefore shows an
explicit out-of-domain warning for likely hospitality/service reviews.

There is a cost to broader coverage: on the legacy 1,962-row clothing test, the
new model scores 0.049740 ROUGE-1 versus 0.114917 for the previous specialist.
Complete results and examples are stored in `artifacts/evaluation.json`. ROUGE
measures lexical overlap, not factuality, usefulness, or fluency.

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

If Community Cloud reports that `PreprocessingConfig` received unexpected
fields immediately after an artifact upgrade, the running process has mixed an
older imported module with newer artifacts. From the app, select **Manage app →
⋮ → Reboot app** to force a clean checkout and import. The pinned dependency
file is also changed when the artifact schema changes so Cloud initiates a clean
environment rebuild automatically.

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
