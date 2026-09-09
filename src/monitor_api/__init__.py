"""Monitor web del pipeline DFT (GPAW/BUHO).

`__version__` es la única fuente de verdad de la versión: la lee
`pyproject.toml` (hatch), la anuncia la app de FastAPI, la devuelve
`GET /api/health`, la muestra el frontend y la comprueba la CI contra el tag
antes de publicar.
"""

__version__ = "0.8.0"
