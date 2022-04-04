#!/bin/bash
# coding: utf-8
set -o xtrace
echo "cloning clip and stuff..."
git clone https://github.com/openai/CLIP
git clone https://github.com/CompVis/taming-transformers
#git clone https://github.com/crowsonkb/v-diffusion-pytorch
echo "downloading this other stuff"
python3.9 -m pip install ftfy regex tqdm omegaconf pytorch-lightning torch
python3.9 -m pip install kornia
python3.9 -m pip install einops
#python3.9 -m pip install pillow==7.1.2
python3.9 -m pip install imageio-ffmpeg 
python3.9 -m pip install torchvision
python3.9 -m pip install imageio
python3.9 -m pip install redis requests TwitterAPI
python3.9 -m pip install 'psycopg[pool, binary]'
