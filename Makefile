SHELL := /usr/bin/env bash

.PHONY: setup run clean report-docx

setup:
	python -m pip install -r requirements.txt

run:
	jupyter nbconvert --to notebook --execute notebooks/03_compression_alpha_pruning.ipynb --output notebooks/03_compression_alpha_pruning.out.ipynb

clean:
	find reports -type f -name "*.png" -delete || true
	find reports -type f -name "*.csv" -delete || true
	rm -f notebooks/03_compression_alpha_pruning.out.ipynb || true

report-docx:
	@if ! command -v pandoc >/dev/null 2>&1; then \
		echo "Pandoc no está instalado. Instálalo desde https://pandoc.org/install.html"; \
		exit 1; \
	fi
	pandoc reports/FINAL_REPORT.md -o reports/FINAL_REPORT.docx --from gfm --toc --metadata title="Proyecto Fourier 2D"


