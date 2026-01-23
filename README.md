# Perch Hoplite

![CI](https://github.com/google-research/perch-hoplite/actions/workflows/ci_uv.yml/badge.svg)

> **Note:** Hoplite is currently going through a major API redesign and some
> parts are still moving.
>
> For non-critical work or if you want to see what's coming, try the latest code
> from this repo.
>
> If you've used Hoplite before, make sure you only install `perch-hoplite<1.0`
> for now, and wait for this note to be removed. We are aiming to release a
> stable version 1.0.0 in January 2026.

Hoplite is a system for storing large volumes of embeddings from machine
perception models. We focus on combining vector search with active learning
workflows, aka [agile modeling](https://arxiv.org/abs/2505.03071).

In brief, agile modeling is a process for rapidly developing classifiers using
embeddings from a pre-trained 'foundation' model. For bioacoustics work, we
find that new classifiers can often be developed for new signals in under
an hour.

**How does it work?**

We first use a bioacoustics model to convert your unlabeled audio data into
embeddings - these are like semantic 'fingerprints' of 5-second audio clips.
Then, you can *search* the embeddings of your data by providing an example of
what you're looking for. You then give feedback on the results - which examples
are and are not what you're looking for. From this feedback, we can quickly
train a classifier. You can then improve on the classifier with
*active learning*: Examine the classifier outputs, provide more feedback, and
re-train the classifier.

A key feature of this workflow is that we pre-compute the embeddings. This
may take a while if you have a large amount of data, but the subsequent search
and classifier training is very efficient.

To get started, load up the following Colab/Jupyter notebooks:

* [`agile/01_embed_audio.ipynb`](perch_hoplite/agile/01_embed_audio.ipynb)
  – Computes embeddings of your audio data.
* [`agile/02_agile_modeling.ipynb`](perch_hoplite/agile/02_agile_modeling.ipynb)
  – Performs search, classification, and active learning.

## Repository Contents

This repository consists of four sub-libraries:

* `db` – The core database functionality for storing embeddings and related
  metadata. The database also handles labels applied to embeddings and vector
  search, both exact and approximate.
* `agile` – Tooling (and example notebooks) for agile modeling on top of the
  Hoplite db layer, combining search and active learning approaches. This
  library includes organizing labeled data and training linear classifiers over
  embeddings, as well as tooling for embedding large datasets.
* `zoo` – A bioacoustics model zoo. A basic wrapper class is provided, and any
  model which can transform windows of audio samples into embeddings can then
  be used in the agile modeling workflow.
* `taxonomy` – A database of taxonomic information, especially for handling
  conversions between the various bird taxonomies.

Each sub-library has its own documentation.

## Installation

We recommend using `uv` or `pip` for installation. `uv` is a fast rust-based
pip-compatible package installer and resolver.

First, install system dependencies for audio processing:
```bash
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg
```

### With `uv`

If you don't have `uv`, you can install it via `pipx install uv` or
`pip install uv`.
If you are developing locally, clone the repository and install in editable
mode:
```bash
git clone https://github.com/google-research/perch-hoplite.git
cd perch-hoplite
uv pip install -e .
```

### With `pip`

You can install the latest stable release from PyPI:
```bash
pip install perch-hoplite
```
Or install the latest version from GitHub:
```bash
pip install git+https://github.com/google-research/perch-hoplite.git
```

After installation, you can run the tests to check that everything is working:
```bash
python -m unittest discover -s perch_hoplite/db/tests -p "*test.py"
python -m unittest discover -s perch_hoplite/taxonomy -p "*test.py"
python -m unittest discover -s perch_hoplite/zoo -p "*test.py"
python -m unittest discover -s perch_hoplite/agile/tests -p "*test.py"
```

### Notes on Dependencies

For GPU support, you can install GPU-enabled tensorflow instead of the default
CPU version *before* installing `perch-hoplite`. See tensorflow documentation
for instructions relative to your CUDA version.

The `zoo` library contains wrappers for various bioacoustic models. Some of
these require JAX. To install with JAX dependencies:
```bash
uv pip install -e '.[jax]'
```
or with pip:
```bash
pip install 'perch-hoplite[jax]'
```

## Disclaimer

This is not an officially supported Google product. This project is not eligible
for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
