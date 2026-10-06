# Convenience targets (Linux/macOS/Codespaces). On Windows, run the python commands directly (see README).
PY ?= python

.PHONY: data features train evaluate app test

data:
	$(PY) -m src.download

features:
	$(PY) -m src.dataset

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

app:
	$(PY) app/app.py

test:
	$(PY) -m pytest -q
