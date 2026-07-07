.PHONY: format style test

# reviewbot needs Python >= 3.10 (pyproject requires-python). Bare `python3` is
# 3.9 on stock macOS, which silently builds a broken venv — so prefer an
# explicit 3.1x if present, and hard-fail below if the resolved one is too old.
PYTHON ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

$(VENV)/.installed: pyproject.toml
	@$(PYTHON) -c 'import sys; sys.exit(sys.version_info[:2] < (3, 10))' || { \
	  echo "Error: $(PYTHON) is $$($(PYTHON) -V 2>&1); reviewbot needs Python >= 3.10."; \
	  echo "Install it (e.g. 'brew install python@3.10') or run 'make PYTHON=python3.10 ...'."; \
	  exit 1; }
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

