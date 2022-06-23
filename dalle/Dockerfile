#FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
pip install --upgrade requests "jax[cuda]" "jaxlib[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \
  && pip install -q \
  git+https://github.com/borisdayma/dalle-mini.git \
  git+https://github.com/patil-suraj/vqgan-jax.git

RUN wandb login --anonymously

COPY ./4chan_dalle.py ./dalle.py
#RUN python3.10 dalle.py
COPY ./postgres_jobs.py ./config.py ./
RUN mkdir /output

ENTRYPOINT ["/usr/bin/python3.8", "/dalle.py"]
