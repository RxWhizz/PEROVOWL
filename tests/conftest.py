"""Configuracion comun de la suite.

Las pruebas no pueden depender de como este montada la maquina que las corre.
Eso vale doble ahora que las pruebas bloquean una publicacion: un fallo que
depende de si este portatil tiene WSL con un entorno de matgl no dice nada sobre
el codigo, y en un runner de CI --- o en el portatil de otro--- daria distinto.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_buscar_mlff_en_esta_maquina():
    """La seleccion de fase no sale a buscar un MLFF durante las pruebas.

    Preparar un trabajo DFT elige ahora la fase con el potencial interatomico, y
    resolver donde vive ese potencial acaba arrancando WSL en Windows: minutos
    de espera en pruebas que no van de eso, y un resultado distinto segun la
    maquina.

    El interruptor solo apaga la BUSQUEDA. Las pruebas que si ejercitan la
    seleccion --- `tests/test_selector_fases.py`--- inyectan su propio runtime y
    siguen pasando por todo el camino.
    """
    previo = os.environ.get("BUHO_FASES")
    os.environ["BUHO_FASES"] = "off"
    yield
    if previo is None:
        os.environ.pop("BUHO_FASES", None)
    else:
        os.environ["BUHO_FASES"] = previo
