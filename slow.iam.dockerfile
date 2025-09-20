# syntax=docker/dockerfile:experimental
FROM python:3.13-slim

##
RUN apt update && apt upgrade -y
RUN apt install -y make curl unzip

##
WORKDIR /slow.iam

##
ARG MODEL_PATH=https://download.pytorch.org/models/resnet18-f37072fd.pth
ARG TORCHVISION_HASH=fab1188

##
ENV TORCH_HOME=/slow.iam/_torch

##
COPY _torch* /slow.iam/_torch/
RUN test -d /slow.iam/_torch/hub/checkpoints || \
    (echo "no vision or model, downloading" && \
    mkdir -pv /slow.iam/_torch/hub/checkpoints && \
    curl -Ls $MODEL_PATH -o /slow.iam/_torch/hub/checkpoints/`basename $MODEL_PATH` && \
    ls -l /slow.iam/_torch/hub/checkpoints && \
    curl -Ls https://github.com/pytorch/vision/zipball/$TORCHVISION_HASH -o /slow.iam/_torch/hub/tv_ball.zip && \
    ls -l /slow.iam/_torch/hub/tv_ball.zip && \
    unzip -q /slow.iam/_torch/hub/tv_ball.zip -d /slow.iam/_torch/hub && \
    mv /slow.iam/_torch/hub/pytorch-vision-* /slow.iam/_torch/hub/pytorch_vision_main ) && \
    find /slow.iam/_torch/ -maxdepth 3 -type d -or -iname '*.pth'

##
COPY requirements*txt Makefile /slow.iam/
RUN --mount=type=cache,mode=0755,id=pip-cache,target=/var/pip-cache \
    PIP_CACHE_DIR=/var/pip-cache \
    make install-docker

##
COPY iklssfr /slow.iam/iklssfr
COPY ireqs /slow.iam/ireqs
COPY ihelp /slow.iam/ihelp
COPY toolz /slow.iam/toolz

CMD ["make", "demo-no-show"]
