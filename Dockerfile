FROM nvidia/cuda:12.8.0-base-ubuntu24.04

RUN apt-get install -y python3.13
RUN apt-get install -y python3-pip

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*
RUN mkdir -p examples/annotations && \
    curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip && \
    unzip annotations_trainval2017.zip -d examples/annotations && \
    rm annotations_trainval2017.zip

RUN pip install .
RUN ls -la


CMD ["/bin/bash"]
