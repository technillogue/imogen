FROM ubuntu:hirsute as libbuilder
WORKDIR /app
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
#RUN add-apt-repository universe && apt update
RUN DEBIAN_FRONTEND="noninteractive" apt update && apt install -yy python3  python3-venv pipenv git
RUN git clone https://github.com/openai/CLIP                 
RUN git clone https://github.com/CompVis/taming-transformers 
RUN python3.9 -m venv /app/venv
COPY pyproject.toml poetry.lock requirements.txt /app/
RUN VIRTUAL_ENV=/app/venv pipenv install -r requirements.txt

FROM ubuntu:hirsute
WORKDIR /app
RUN mkdir -p /app/steps
RUN ln --symbolic --force --no-dereference /usr/share/zoneinfo/EST && echo "EST" > /etc/timezone
RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3 
#RUN apt-get clean autoclean && apt-get autoremove --yes && rm -rf /var/lib/{apt,dpkg,cache,log}/

COPY --from=libbuilder /app/venv/lib/python3/site-packages /app/taming-transformers /app/CLIP /app/
COPY ./ganclip_functional.py ./finished_prompts.json ./prompts.json  /app/ 
ENTRYPOINT ["/usr/bin/python3", "/app/ganclip_functional.py"]
