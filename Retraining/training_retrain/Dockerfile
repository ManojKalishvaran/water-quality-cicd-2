FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "main/retrain.py"]