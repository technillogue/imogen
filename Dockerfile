# FROM ubuntu:hirsute as libbuilder
# WORKDIR /app
# RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
# #RUN add-apt-repository universe && apt update
# RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -yy python3  python3-venv pipenv git
# RUN git clone https://github.com/openai/CLIP                 
# RUN git clone https://github.com/CompVis/taming-transformers 
# RUN python3.9 -m venv /app/venv
# # this isn't a thing lol
# COPY pyproject.toml poetry.lock requirements.txt /app/
# RUN VIRTUAL_ENV=/app/venv pipenv install -r requirements.txt

FROM ubuntu:hirsute
WORKDIR /app
RUN mkdir -p /app/steps
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y python3 python3-pip ffmpeg git curl
COPY ./install_deps.sh .
RUN ./install_deps.sh
COPY ./download_modals.sh .
RUN ./download_modals.sh
COPY ./utils.py ./better_imagegen.py ./mk_video.py ./postgres_jobs.py /app/ 
ENTRYPOINT ["/usr/bin/python3", "/app/postgres_jobs.py"]
