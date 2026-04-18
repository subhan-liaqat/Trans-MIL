#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y openslide-tools libopenslide-dev

python -m pip install --upgrade pip
python -m pip install \
  addict==2.4.0 \
  albumentations==1.4.10 \
  einops==0.8.0 \
  h5py==3.11.0 \
  matplotlib==3.8.4 \
  numpy==1.26.4 \
  nystrom-attention==0.0.14 \
  omegaconf==2.3.0 \
  opencv-python-headless==4.10.0.84 \
  openslide-python==1.4.1 \
  pandas==2.2.2 \
  Pillow==10.3.0 \
  pytorch-lightning==1.9.5 \
  PyYAML==6.0.1 \
  timm==0.9.16 \
  torchmetrics==1.3.2
