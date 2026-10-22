# One command per step. Run from the project folder (macOS / Linux).
PYTHON = .venv/bin/python

.PHONY: setup test app all download dataset train evaluate

setup:     ## Python 3.13 environment and libraries (needs uv: https://docs.astral.sh/uv/)
	uv python install 3.13
	uv venv --python 3.13 --seed .venv
	$(PYTHON) -m pip install -r requirements.txt

test:      ## all tests (a few seconds)
	$(PYTHON) -m pytest

app:       ## the website on http://localhost:8501 (uses the included model)
	.venv/bin/streamlit run app/app.py

download:  ## stream 250,000 blitz games from Lichess (about 20 min)
	$(PYTHON) -m src.download

dataset:   ## features + train/test table (about 15 min)
	$(PYTHON) -m src.dataset

train:     ## compare models, tune, save models/model.joblib (about 10 min)
	$(PYTHON) -m src.train

evaluate:  ## test-set results, tables and figures (about 3 min)
	$(PYTHON) -m src.evaluate

all: download dataset train evaluate  ## reproduce every result (about 1 hour)
