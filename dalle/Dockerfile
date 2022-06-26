FROM python:3.10 as libbuilder
WORKDIR /app
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update 
RUN apt-get -y install git
#RUN pip install poetry
RUN python3.10 -m venv /app/venv 
#ENV PIP_FIND_LINKS=https://download.pytorch.org/whl/cu113/torch_stable.html
#COPY ./pyproject.toml /app/
 #./poetry.lock /app/
ENV VIRTUAL_ENV=/app/venv 
RUN /app/venv/bin/pip install --upgrade pip "jax[cuda]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN /app/venv/bin/pip install "psycopg[pool, binary]" TwitterAPI requests
RUN /app/venv/bin/pip install -q \
    git+https://github.com/borisdayma/dalle-mini.git \
    git+https://github.com/patil-suraj/vqgan-jax.git 

FROM python:3.10 as models
WORKDIR /app
RUN pip install wandb transformers flax
RUN wandb login --anonymously
COPY ./download_models.py /app/
RUN python3.10 download_models.py

# FROM curlimages/curl as curl
# USER root
# WORKDIR /app
# RUN pwd
# RUN curl -sSL https://install.python-poetry.org > /app/install.py

FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND="noninteractive"
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt-get update && apt-get install -y software-properties-common \ 
  && add-apt-repository ppa:deadsnakes/ppa \ 
  && apt-get install -y python3.10 python3.10-venv git \
  && apt-get remove -y python3.8 python3-minimal \
  && apt-get autoremove -y \
  && rm -rf /var/lib/{apt,dpkg,cache,log}/
COPY --from=models /root/.cache/wandb /root/.cache/wandb
#RUN python3.10 -m ensurepip
# COPY --from=curl /app/install.py /install.py 
# RUN python3.10 /install.py --version 1.2.0b2
# WORKDIR /app
# COPY ./pyproject.toml /app
# RUN poetry install 
COPY --from=libbuilder /app/venv/lib/python3.10/site-packages /app/
# RUN python3.10 -m pip install --upgrade pip "jax[cuda]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \ 
#   && python3.10 -m pip install -q \
#     git+https://github.com/borisdayma/dalle-mini.git \
#     git+https://github.com/patil-suraj/vqgan-jax.git \
#   && python3.10 -m pip install "psycopg[pool, binary]" TwitterAPI requests

WORKDIR /app
RUN python3.10 -m wandb login --anonymously
RUN mkdir /app/output
COPY ./dalle.py /app/
RUN python3.10 /app/dalle.py
COPY ./postgres_jobs.py ./config.py  /app/
ENTRYPOINT ["/usr/bin/python3.10", "/app/postgres_jobs.py"]
