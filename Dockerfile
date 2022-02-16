FROM ubuntu:hirsute as libbuilder
WORKDIR /app
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install -yy python3.9 python3.9-venv pipenv
RUN python3.9 -m venv /app/venv
ENV PIP_FIND_LINKS=https://download.pytorch.org/whl/cu113/torch_stable.html
COPY Pipfile.lock Pipfile /app/
RUN VIRTUAL_ENV=/app/venv pipenv install 

FROM ubuntu:hirsute
WORKDIR /app
RUN mkdir -p /app/steps
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y python3 python3-pip ffmpeg git curl
RUN git clone https://github.com/openai/CLIP && git clone https://github.com/CompVis/taming-transformers
COPY ./download_modals.sh .
RUN ./download_modals.sh
COPY --from=libbuilder /app/venv/lib/python3.9/site-packages /app/
COPY ./utils.py ./better_imagegen.py ./mk_video.py ./postgres_jobs.py /app/ 
ENTRYPOINT ["/usr/bin/python3", "/app/postgres_jobs.py"]
