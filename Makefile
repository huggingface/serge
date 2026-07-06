.PHONY: format style test

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

$(VENV)/.installed: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e '.[web]' pytest ruff
	touch $(VENV)/.installed

format: $(VENV)/.installed
	$(RUFF) format reviewbot tests

style: format
	$(RUFF) check --fix reviewbot tests

test: $(VENV)/.installed
	$(VENV_PYTHON) -m pytest tests/

