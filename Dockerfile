FROM nvidia/cuda:12.8.0-base-ubuntu24.04

RUN apt update
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app


RUN pip install . --break-system-packages
RUN ls -la


CMD ["/bin/bash"]
