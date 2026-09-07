"""BUHO: pipeline de descubrimiento de perovskitas ABX3.

Genera composiciones, las criba con una cascada de tres tiers (física,
surrogate de bandgap, MLFF de estabilidad), calcula con DFT solo las que
sobreviven y reentrena el surrogate con lo calculado.

Todo es accesible por comandos:

    buho doctor                 # entorno, datasets PAW, datos y modelos
    buho generate generate      # componer candidatos
    buho screening run          # cribar
    buho dft-jobs prepare-relax # preparar los calculos
    buho active-learning advance

La version sale de los metadatos de la distribucion instalada, que hatch toma
de src/monitor_api/__init__.py. Tenerla escrita aqui a mano la dejaba clavada
en 0.1.0 mientras el resto del proyecto ya iba por 0.2.0, y `buho --version`
mentia.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("buho")
except PackageNotFoundError:  # ejecutado desde el repo, sin instalar
    __version__ = "0.7.3"
