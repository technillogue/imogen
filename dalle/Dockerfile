# FROM curlimages/curl as curl
# USER root
# WORKDIR /app
# RUN pwd
# RUN curl -sSL https://install.python-poetry.org > /app/install.py

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND="noninteractive"
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt-get update && apt-get install -y software-properties-common \ 
  && add-apt-repository ppa:deadsnakes/ppa \ 
  && apt-get install -y python3.10 python3.10-venv git \
  && apt-get remove -y python3.8 python3-minimal \
  && apt-get autoremove -y \
  && rm -rf /var/lib/{apt,dpkg,cache,log}/
RUN python3.10 -m ensurepip
# COPY --from=curl /app/install.py /install.py 
# RUN python3.10 /install.py --version 1.2.0b2
# WORKDIR /app
# COPY ./pyproject.toml /app
# RUN poetry install 
RUN python3.10 -m pip install --upgrade pip "jax[cuda]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \ 
  && python3.10 -m pip install -q \
    git+https://github.com/borisdayma/dalle-mini.git \
    git+https://github.com/patil-suraj/vqgan-jax.git \
  && python3.10 -m pip install "psycopg[pool, binary]" TwitterAPI requests

RUN wandb login --anonymously
WORKDIR /app
COPY ./4chan_dalle.py ./dalle.py ./postgres_jobs.py ./config.py 
RUN mkdir /output
ENTRYPOINT ["/usr/bin/python3.8", "/dalle.py"]
