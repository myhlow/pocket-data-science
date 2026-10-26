#!/usr/bin/env bash
# Pocket Data Science: Termux PRoot Environment Bootstrapper
# Team: Pocket Data Lab

set -e

echo "=== [1/4] Updating APT Repositories ==="
apt-get update && apt-get upgrade -y

echo "=== [2/4] Installing Native Compilers & BLAS Libraries ==="
apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libomp-dev \
    cmake \
    git \
    curl \
    jq

echo "=== [3/4] Installing Edge Machine Learning Stack ==="
pip3 install --upgrade pip setuptools wheel
pip3 install \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    lightgbm \
    catboost \
    kaggle

echo "=== [4/4] Verifying Environment Installation ==="
python3 -c "
import numpy, scipy, pandas, sklearn, lightgbm, catboost
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('Pandas:', pandas.__version__)
print('Scikit-Learn:', sklearn.__version__)
print('LightGBM:', lightgbm.__version__)
print('CatBoost:', catboost.__version__)
print('All core ML packages loaded successfully on ARM64!')
"
echo "=== Bootstrap Complete! Ready for Pocket Data Science Pipelines ==="
