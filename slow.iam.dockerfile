# syntax=docker/dockerfile:experimental
FROM python:3.12-slim

WORKDIR /slow.iam

COPY  requirements*txt Makefile /slow.iam/
COPY  iklssfr /slow.iam/iklssfr
COPY  toolz /slow.iam/toolz

RUN --mount=type=cache,mode=0755,id=pip-cache,target=/var/pip-cache \
    apt update && apt upgrade -y && apt install -y make && \
    make install

CMD ["make", "demo"]


