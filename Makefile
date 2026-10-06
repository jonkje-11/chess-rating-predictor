# Convenience targets (Linux/macOS/Codespaces). On Windows, run the python commands directly (see README).
PY ?= python

.PHONY: all data features train evaluate eda app test

all: data features train evaluate

data:        ## stream + filter games from Lichess (~1.5 min for 250k games)
	$(PY) -m src.download

features:    ## per-player feature table + grouped train/test split (~1.5 min)
	$(PY) -m src.dataset

train:       ## compare models, tune the best, save models/model.joblib
	$(PY) -m src.train

evaluate:    ## figures and tables for the report
	$(PY) -m src.evaluate

eda:         ## re-run the EDA notebook in place
	cd notebooks && jupyter execute --inplace 01_eda.ipynb

app:         ## run the web app on http://127.0.0.1:7860
	$(PY) app/app.py

test:
	$(PY) -m pytest -q
