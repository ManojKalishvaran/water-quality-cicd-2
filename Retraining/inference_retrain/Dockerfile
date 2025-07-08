FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

# ARG MODEL_CONTAINER=trained-models

# ENV MODEL_CONTAINER=${MODEL_CONTAINER}

CMD python main/scoring.py 

# FROM python:3.10-slim

# WORKDIR /app
# COPY . .

# RUN pip install -r requirements.txt

# ENV MODEL_CONTAINER=trained-models 

# RUN echo "✅ MODEL_CONTAINER set to: $MODEL_CONTAINER"

# CMD python main/scoring.py


# FROM python:3.10-slim

# WORKDIR /app
# COPY . .

# # Install requirements
# RUN pip install -r requirements.txt

# # Accept build-time argument
# ARG MODEL_CONTAINER

# # Set it as a runtime environment variable
# ENV MODEL_CONTAINER=$MODEL_CONTAINER

# RUN echo "✅ MODEL_CONTAINER at build: ${MODEL_CONTAINER}"

# CMD ["python", "main/scoring.py"]
