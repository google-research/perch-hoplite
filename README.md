# Perch Hoplite

![CI](https://github.com/google-research/perch-hoplite/actions/workflows/ci_pip.yml/badge.svg)

Hoplite is a system for storing large volumes of embeddings from machine
perception models. We focus on combining vector search with active learning
workflows, aka [agile modeling](https://arxiv.org/abs/2505.03071).

In brief, agile modeling is a process for rapidly developing classifiers using
embeddings from a pre-trained 'foundation' model. For bioacoustics work, we
find that new classifiers can often be developed for new signals in under
an hour.

How does it work?

We first use a bioacoustics model to convert your unlabeled audio data into
embeddings - these are like semantic 'fingerprints' of 5-second audio clips.
Then, you can *search* the embeddings of your data by providing an example
of what you're looking for. You then give feedback on the results - which
examples are and are not what you're looking for. From this feedback, we
can quickly train a classifier. You can then improve on the classifier with
*active learning*: Examine the classifier outputs, provide more feedback,
and re-train the classifier.

A key feature of this workflow is that we pre-compute the embeddings. This
may take a while if you have a large amount of data, but the subsequent
search and classifier training is very efficient.

To get started, load up the following Colab/Jupyter notebooks:
* `agile/1_embed_audio_v2.ipynb` - Computes embeddings of your audio data.
* `agile/2_agile_modeling_v2.ipynb` - Perform search, classification, and active
  learning.

## Repository Contents

This repository consists of four sub-libraries:

* `db` - The core database functionality for storing embeddings and related
metadata. The database also handles labels applied to embeddings and vector
search, both exact and approximate.
* `agile` - Tooling (and example notebooks) for agile modeling on top of the
Hoplite db layer, combining search and active learning approaches.
This library includes organizing labeled data and training linear
classifiers over embeddings, as well as tooling for embedding large datasets.
* `zoo` - A bioacoustics model zoo. A basic wrapper class is provided, and
any model which can transform windows of audio samples into embeddings
can then be used in the agile modeling workflow.
* `taxonomy` - A database of taxonomic information, especially for handling
conversions between the various bird taxonomies.

Each sub-library has its own documentation.

# Installation

The repository can be installed with either pip or poetry. Poetry allows more
granular management of dependencies.

First, install some basic dependencies. Note that for GPU support, you may
install `tensorflow[and-cuda]` instead of `tensorflow-cpu`.

```bash
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg
pip install absl-py
pip install requests
# You may skip tensorflow installation if only using the hoplite/db library.
# However, these are required for agile modeling and most models in the zoo.
pip install tensorflow-cpu
pip install tensorflow-hub
```

Then to install with pip:
```bash
pip install git+https://github.com/google-research/perch-hoplite.git
```

Then run the tests and check that they pass:
```bash
python -m unittest discover -s perch_hoplite/db/tests -p "*test.py"
python -m unittest discover -s perch_hoplite/taxonomy -p "*test.py"
python -m unittest discover -s perch_hoplite/zoo -p "*test.py"
python -m unittest discover -s perch_hoplite/agile/tests -p "*test.py"
```

Or, install with poetry:
```bash
# Install Poetry for package management
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies specified in the poetry configs.
poetry install
```

## Notes on Dependencies

Machine learning framework libraries are pretty heavy! It can also be difficult to coordinate CUDA versions across multiple frameworks to ensure good GPU behavior.  Thus, we provide some ability to select dependencies according to your needs.

Tensorflow is used in the `agile` library for training linear classifiers. If you do not need the `agile` library or any of the tensorflow models in the `zoo`, you may skip installation of tensorflow dependencies with pip. Alternatively, you can use poetry to install without tensorflow like so:

```bash
poetry install --without tf
```

The primary place where multiple frameworks may be needed is in the `zoo` library, which provides wrappers for various bioacoustic models. To install with JAX (allowing use of some models in the `zoo`):

```bash
poetry install --with jax
```

# Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
