#!/bin/bash
#
# Generate requirements.txt files from pyproject.toml using uv.

set -e

PYTHON_VERSION=${1:-3.10}
echo "Installing python $PYTHON_VERSION"
uv python install $PYTHON_VERSION

echo "Generating requirements for Python version: $PYTHON_VERSION"

echo "Generating requirements.txt..."
uv pip compile --python-version $PYTHON_VERSION pyproject.toml -o requirements.txt --generate-hashes

echo "Generating requirements-jax.txt..."
uv pip compile --python-version $PYTHON_VERSION pyproject.toml --extra jax -o requirements-jax.txt --generate-hashes

echo "Done."
