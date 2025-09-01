MAKEFILE = $(realpath $(firstword $(MAKEFILE_LIST)))
BASE_DIR ?= $(shell dirname $(MAKEFILE))

VENV ?= $(BASE_DIR)/_venv
PYTHON_SYSTEM ?= $(shell which python3)
PYTHON ?= "${VENV}/bin/python"
IPYTHON ?= "${VENV}/bin/ipython"
PIP ?= "${VENV}/bin/pip" --disable-pip-version-check --require-virtualenv --retries 1

DOCKER ?= DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker
DOCKER_IMAGE_NAME ?= slow-iam
DOCKER_CONTAINER_NAME ?= slow-iam

DATA_DIR ?= $(BASE_DIR)/data

SOURCE_DIRS ?= iklssfr ihelp toolz

##
.PHONY: help tea install install-extra install-dev shell format lint
.PHONY: docker-build-run docker-build-image docker-container-run
.PHONY: demo demo-no-show demo-run demo-generate-train-set demo-generate-test-set demo-train-model demo-run-model


help:
	@egrep "(^|  )## " ${MAKEFILE} | sed 's/:.*##/ ##/' | sed 's/ ## / -- /'
	@echo "> make demo"

tea: install install-dev format lint demo  ## install and run demo

## env
$(PYTHON):  ## create venv
	[ -d $(VENV) ] || $(PYTHON_SYSTEM) -m venv "$(VENV)" --clear

install: $(PYTHON)  ## install requirements
	$(PIP) install --upgrade pip setuptools
	$(PIP) install --requirement requirements.txt --log ${VENV}/pip.log

install-extra: $(PYTHON)  ## install extra requirements
	-$(PIP) install --requirement requirements-extra.txt --log ${VENV}/pip.log 2>&1

install-dev: install  ## install dev tools
	$(PIP) install --requirement requirements-dev.txt


## docker
docker-build-run: docker-build-image docker-container-run  ## build and run docker container

docker-build-image:  ## build docker image
	$(DOCKER) build -t $(DOCKER_IMAGE_NAME) -f slow.iam.dockerfile .

docker-container-run:  ## run docker container
	$(DOCKER) run \
	--interactive=true --tty=true --rm=true --net none \
	--name=$(DOCKER_CONTAINER_NAME) \
	$(DOCKER_IMAGE_NAME)


## shell
$(IPYTHON):
	test -f $(IPYTHON) || make install-dev


shell: $(IPYTHON)  ## run ipython shell
	$(IPYTHON) -c 'import re, math, json; import torch; print(f"{torch.version.__version__=}")' -i

format:  ## format
	$(PYTHON) -m isort $(SOURCE_DIRS)
	$(PYTHON) -m ruff format $(SOURCE_DIRS)

lint:  ## lint
	$(PYTHON) -m black --check $(SOURCE_DIRS)
	$(PYTHON) -m ruff check $(SOURCE_DIRS)

## demo
demo: DEMO_RUN_MODEL_OPTS ?= --show --show-original --limit 60 --summarize
demo: DEMO_TRAIN_MODEL_OPTS ?= --model resnet18 --epochs-limit 17 --batch-size 8 --data-autocontrast --data-normalize --workers 1
demo: demo-run  ## run demo

demo-no-show: DEMO_RUN_MODEL_OPTS = --no-show --limit 100 --summarize
demo-no-show: demo-run  ## run demo without show

demo-run: demo-generate-train-set demo-generate-test-set demo-train-model demo-run-model

demo-generate-train-set:
	# [1] generate train dataset
	[ -d $(DATA_DIR)/o34/train/ ] || for fg in o 3 4 5 o/4 o/3; do \
	$(PYTHON) toolz/o34.py \
	--save-path "$(DATA_DIR)/o34/train/$${fg}" \
	--count 199 \
	--figures-count 21 \
	--figures "$${fg}"; \
	done

demo-generate-test-set:
	# [2] generate test dataset
	[ -d $(DATA_DIR)/o34/test/ ] || for fg in o 3 4 5 o3 o4; do \
	$(PYTHON) toolz/o34.py \
	--save-path "$(DATA_DIR)/o34/test/$${fg}" \
	--count 9 \
	--figures-count 17 \
	--figures "$${fg}"; \
	done

demo-train-model:
	# [3] train model
	[ -f ""$(DATA_DIR)/o34/model.pth"" ] || $(PYTHON) -u \
	-m iklssfr.train \
	$(DEMO_TRAIN_MODEL_OPTS) \
	--model-path "$(DATA_DIR)/o34/model.pth" \
	--log "$(DATA_DIR)/o34/model.log" \
	"$(DATA_DIR)/o34/train/"

demo-run-model:
	# [4] run model
	$(PYTHON) -u \
	-m iklssfr.run \
	--model-path "$(DATA_DIR)/o34/model.pth" \
	$(DEMO_RUN_MODEL_OPTS) \
	"$(DATA_DIR)/o34/test/"
