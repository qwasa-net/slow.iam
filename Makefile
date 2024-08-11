BASE_DIR ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VENV ?= $(BASE_DIR)/.venv
PYTHON_SYSTEM ?= $(shell which python3)
PYTHON ?= "${VENV}/bin/python"
PIP ?= "${VENV}/bin/pip"

DATA_DIR ?= $(BASE_DIR)/data

env:
	[ -d $(VENV) ] || $(PYTHON_SYSTEM) -m venv $(VENV) --clear

install: env
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt


install-dev: install
	$(PIP) install -r requirements-dev.txt


demo: install 

	# [1] generate train dataset
	[ -d $(DATA_DIR)/o34/train/ ] || for fg in o 3 4 o4 o3; do \
	$(PYTHON) toolz/o34.py \
	--save-path "$(DATA_DIR)/o34/train/$${fg}" \
	--count 98 \
	--figures-count 25 \
	--figures "$${fg}"; \
	done

	# [2] generate test dataset
	[ -d $(DATA_DIR)/o34/test/ ] || for fg in o 3 4 o4 o3; do \
	$(PYTHON) toolz/o34.py \
	--save-path "$(DATA_DIR)/o34/test/$${fg}" \
	--count 10 \
	--figures-count 15 \
	--figures "$${fg}"; \
	done

	# [3] train model
	$(PYTHON) -u \
	iklssfr/train.py \
	--data "$(DATA_DIR)/o34/train/" \
	--epochs-limit 25 \
	--data-autocontrast \
	--data-normalize \
	--model-path "$(DATA_DIR)/o34/model.pth" \
	--log "$(DATA_DIR)/o34/model.log"

	# [4] run model
	$(PYTHON) -u \
	iklssfr/run.py \
	--data "$(DATA_DIR)/o34/test/" \
	--model-path "$(DATA_DIR)/o34/model.pth" \
	--show \
	--limit 50
