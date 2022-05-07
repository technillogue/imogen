FROM python:3.9 as libbuilder
WORKDIR /app
RUN pip install poetry
RUN apt-get update && apt-get install -yy git
RUN python3.9 -m venv /app/venv 
COPY ./pyproject.toml /app/
RUN VIRTUAL_ENV=/app/venv poetry install 

FROM scratch AS model
# put it seperately in hopes of appeasing the cache gods
COPY ./reaction_predictor.pth / 

FROM scratch AS prompts
# put it seperately in hopes of appeasing the cache gods
COPY ./prompts.pth / 

FROM ubuntu:hirsute
WORKDIR /app
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y python3
COPY --from=libbuilder /app/venv/lib/python3.9/site-packages /app/
COPY --from=model /reaction_predictor.pth /app/reaction_predictor.pth
COPY --from=prompts /prompts.pth /app/prompts.pth
COPY ./server.py ./core.py ./knn.py /app/ 
ENTRYPOINT ["/usr/bin/python3", "/app/server.py"]
