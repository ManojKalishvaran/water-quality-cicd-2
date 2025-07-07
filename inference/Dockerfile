FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ARG MODEL_CONTAINER=trained-models

ENV MODEL_CONTAINER=${MODEL_CONTAINER}

CMD python main/scoring.py --model_container_name $MODEL_CONTAINER