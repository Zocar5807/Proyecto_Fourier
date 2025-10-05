SHELL := /usr/bin/env bash

.PHONY: setup run clean

setup:
	python -m pip install -r requirements.txt

run:
	jupyter nbconvert --to notebook --execute notebooks/03_compression_alpha_pruning.ipynb --output notebooks/03_compression_alpha_pruning.out.ipynb

clean:
	find reports -type f -name "*.png" -delete || true
	find reports -type f -name "*.csv" -delete || true
	rm -f notebooks/03_compression_alpha_pruning.out.ipynb || true


