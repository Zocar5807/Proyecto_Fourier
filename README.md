# PROYECTO_FOURIER: FFT2D para análisis, filtrado y compresión

Proyecto reproducible en Python/Jupyter para estudiar análisis espectral, filtrado y compresión por poda de coeficientes (α-pruning) en 2D.

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

## Uso rápido

Desde los notebooks, añade el path al proyecto si es necesario:
```python
import sys, os
sys.path.append(os.path.abspath(".."))

from src.compression import global_prune, adaptive_radial_prune
from src.fourier_tools import energy_from_spectrum
from src.metrics import evaluate
```

Ejecutar notebook principal y generar reportes:
```bash
jupyter nbconvert --execute notebooks/03_compression_alpha_pruning.ipynb
```

## Licencia y autor

MIT License. Autor: André (y colaboradores).
