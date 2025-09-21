FROM nvidia/cuda:12.8.0-base-ubuntu24.04

WORKDIR /app

COPY . /app

CMD ["python", "main.py"]