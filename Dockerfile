FROM python:3.9 as libbuilder
WORKDIR /app
RUN pip install poetry
RUN python3.9 -m venv /app/venv 
#ENV PIP_FIND_LINKS=https://download.pytorch.org/whl/cu113/torch_stable.html
COPY ./pyproject.toml /app/
 #./poetry.lock /app/
RUN VIRTUAL_ENV=/app/venv poetry install 

FROM curlimages/curl as models
USER root
WORKDIR /app
RUN curl -L -o vqgan_imagenet_f16_16384.yaml 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' \
 && curl -L -o vqgan_imagenet_f16_16384.ckpt 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' \ 
 && curl -L -o wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt'  \
 && curl -L -o wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml'

FROM ubuntu:hirsute
WORKDIR /app
RUN mkdir -p /app/steps
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y python3 ffmpeg git
COPY --from=models /app/vqgan_imagenet_f16_16384.yaml /app/vqgan_imagenet_f16_16384.ckpt /app/wikiart_16384.ckpt /app/wikiart_16384.yaml /app/
COPY --from=libbuilder /app/venv/lib/python3.9/site-packages /app/
RUN git clone https://github.com/openai/CLIP && git clone https://github.com/CompVis/taming-transformers
COPY ./CHANGELOG.md ./utils.py ./better_imagegen.py ./mk_video.py ./postgres_jobs.py /app/ 
RUN python3.9 better_imagegen.py || true
ENTRYPOINT ["/usr/bin/python3", "/app/postgres_jobs.py"]
