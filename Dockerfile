FROM nvidia/cuda:latest

WORKDIR /app

COPY . /app

CMD ["python", "main.py"]