# PROYECTO_FOURIER: FFT2D para análisis, filtrado y compresión

Proyecto reproducible en Python/Jupyter para estudiar análisis espectral, filtrado y compresión por poda de coeficientes (α-pruning) en 2D.

## Nota para la profesora

El documento del informe final se llama **“Reporte final.pdf”** y está ubicado en la **raíz del repositorio** (`./Reporte final.pdf`). Ese documento describe los objetivos, metodología, formulación matemática, resultados y conclusiones del proyecto de forma detallada. Adicionalmente, se incluyen las figuras y tablas generadas automáticamente en `reports/` y una versión en Markdown del informe en `reports/FINAL_REPORT.md`.

## Estructura

```
proyecto_fourier/
├── notebooks/
│   ├── 00_setup_fft.ipynb
│   ├── 01_energy_integrals.ipynb
│   ├── 02_filtering_reconstruction.ipynb
│   └── 03_compression_alpha_pruning.ipynb
├── src/
│   ├── __init__.py
│   ├── fourier_tools.py
│   ├── filters.py
│   ├── compression.py
│   └── metrics.py
├── reports/
├── diary/
│   ├── compression_notes.md
│   └── IA_log.md
├── data/
│   ├── README.md
│   └── test_images/
├── requirements.txt
├── README.md
└── (opcional) Makefile
```

## Descarga

- Clonar este repositorio:
```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto_fourier
```

- O descargar el ZIP desde la plataforma de entrega y descomprimirlo.

## Instalación

1) Crear/activar entorno (ej. venv local):
```bash
python -m venv venv
venv\Scripts\activate
```

2) Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Verificación rápida

Ejecutar pruebas y el notebook principal (Windows/PowerShell):
```powershell
python -m pytest -q
python -m jupyter nbconvert --to notebook --execute .\notebooks\03_compression_alpha_pruning.ipynb --output 03_compression_alpha_pruning.out --output-dir .\notebooks
```

## Uso rápido

Desde los notebooks, añade el path al proyecto si es necesario:
```python
import sys, os
sys.path.append(os.path.abspath(".."))

from src.compression import global_prune, adaptive_radial_prune
from src.fourier_tools import energy_from_spectrum
from src.metrics import evaluate
```

Ejecutar notebook principal y generar reportes (alternativa):
```bash
jupyter nbconvert --to notebook --execute notebooks/03_compression_alpha_pruning.ipynb --output 03_compression_alpha_pruning.out --output-dir notebooks
```

Las figuras y CSVs se guardan en `reports/`.

## Licencia y autor

MIT License. Autor: André (y colaboradores).
