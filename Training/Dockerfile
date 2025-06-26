# FROM python:3.10-slim

# WORKDIR /main

# COPY . /main/

# RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main/training.py"]