"""Construye el `selector_fase` que el preparador de trabajos DFT espera.

Por que este modulo existe
--------------------------
`buho.structure.fases` sabe elegir la fase de menor energia y esta probado, pero
nadie lo llamaba: `RelaxationJobPreparer` acepta un `selector_fase` y ningun
sitio de produccion se lo pasaba, asi que la rama `if self._selector_fase is
None` se tomaba siempre y el protocolo autonomo preparaba la cubica. Sobre
CsPbI3 eso son 124 meV por formula y ~1 eV de bandgap: la fase alfa cubica solo
existe por encima de 330 C.

Lo que falta para cerrar ese hueco no es cristalografia nueva, sino cruzar una
frontera. `seleccionar_fase` relaja con FIRE sobre un filtro de celda y necesita
un calculador ASE **en proceso**; el potencial vive en el entorno del MLFF, que
en Windows esta dentro de WSL. De ahi el reparto: la seleccion corre en el
worker (`buho_mlff_worker.py --fases`) y de vuelta solo cruza la geometria
ganadora.

Un lote, no N llamadas
----------------------
`_elegir_fase` se invoca candidato a candidato dentro del bucle de `prepare()`.
Cargar el potencial cuesta mucho mas que usarlo --- y en WSL hay que sumar el
arranque de la distro--- asi que una llamada por candidato multiplicaria ese
coste fijo por N. El selector construye los padres de TODOS los candidatos de la
ronda en la primera invocacion, hace una sola llamada y sirve el resto de cache.

Nada de esto puede tumbar una ronda: sin MLFF la fabrica devuelve None y el
preparador usa la cubica, que es lo que habia antes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from buho.mlff_runtime import MLFFUnavailableError, resolve as resolve_mlff
from buho.structure import fases as modulo_fases
from buho.structure.build_abx3 import ABX3StructureBuilder

log = logging.getLogger(__name__)

#: Valores de BUHO_FASES que apagan la busqueda de un MLFF en esta maquina.
_APAGADO = {"0", "off", "false", "no"}


def _config_fases(config: dict) -> dict[str, Any]:
    return ((config or {}).get("discovery", {}) or {}).get("fases", {}) or {}


def _opciones_fases(config: dict) -> dict[str, Any]:
    """Lo que el worker necesita saber para relajar, desde generator.yaml."""
    cfg = _config_fases(config)
    return {clave: cfg[clave]
            for clave in ("modelo_pes", "fmax", "pasos", "supercelda_base")
            if cfg.get(clave) is not None}


class _SelectorPorLote:
    """El callable que recibe `RelaxationJobPreparer`, con su cache."""

    def __init__(self, config: dict, runtime, candidatos: Sequence[Any],
                 *, project_root: Path | None = None):
        self._cfg = config
        self._runtime = runtime
        self._candidatos = list(candidatos)
        self._root = project_root
        self._builder = ABX3StructureBuilder(
            config, random_seed=config.get("random_seed", 42))
        self._opciones = _opciones_fases(config)
        self._cache: dict[str, dict[str, Any]] | None = None

    # -- El lote --------------------------------------------------------------

    def _payload(self) -> list[dict[str, Any]]:
        estructuras: list[dict[str, Any]] = []
        for c in self._candidatos:
            try:
                atoms, meta = self._builder.build(c, out_dir=None, export=False)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: no se pudo construir el padre cubico (%s: %s); "
                            "se quedara con la cubica",
                            c.candidate_id, type(exc).__name__, exc)
                continue
            estructuras.append({
                "candidate_id": c.candidate_id,
                "symbols": list(atoms.get_chemical_symbols()),
                "positions": [[float(v) for v in f] for f in atoms.get_positions()],
                "cell": [[float(v) for v in f] for f in atoms.get_cell()],
                "pbc": [bool(v) for v in atoms.get_pbc()],
                "b_sites": list(c.B_site_species),
                "x_sites": list(c.X_site_species),
                # El tamano lo pone la semilla calibrada y no el potencial: ver
                # el reparto documentado en fases.seleccionar_fase.
                "a_semilla_A": float(meta["lattice_constant_A"]),
            })
        return estructuras

    def _resolver_lote(self) -> dict[str, dict[str, Any]]:
        estructuras = self._payload()
        if not estructuras:
            return {}
        log.info("seleccion de fase: %d estructuras en una sola llamada al MLFF",
                 len(estructuras))
        try:
            resultados = self._runtime.fases(estructuras, opciones=self._opciones)
        except Exception as exc:  # noqa: BLE001
            # El fallo se cachea a proposito: si no, cada candidato volveria a
            # arrancar WSL para recibir el mismo error N veces.
            motivo = f"{type(exc).__name__}: {exc}"
            log.warning("la seleccion de fase fallo para toda la ronda (%s); "
                        "se preparan las cubicas", motivo)
            return {e["candidate_id"]: {"ok": False, "motivo": motivo}
                    for e in estructuras}
        return {str(r.get("candidate_id", "")): r for r in resultados}

    # -- Un candidato ---------------------------------------------------------

    def __call__(self, candidato, atoms, meta) -> dict[str, Any]:
        if self._cache is None:
            self._cache = self._resolver_lote()

        r = self._cache.get(candidato.candidate_id)
        if r is None:
            return {"ok": False, "motivo": "el worker no devolvio esta estructura"}
        if not r.get("ok"):
            return r

        geometria = r.get("geometria")
        if not geometria:
            return {"ok": False, "motivo": "la fase elegida vino sin geometria"}

        from ase import Atoms

        elegida = Atoms(
            symbols=geometria["symbols"],
            positions=geometria["positions"],
            cell=geometria["cell"],
            pbc=geometria.get("pbc", True),
        )

        salida = {k: v for k, v in r.items() if k != "geometria"}
        salida["atoms"] = elegida
        # El grupo espacial se reidentifica aqui: spglib viaja en el binario del
        # motor pero no tiene por que estar en el entorno del MLFF, donde
        # `grupo_espacial` degrada a "no disponible". Es el dato que va al ledger
        # y al informe, asi que se mide donde si se puede.
        local = modulo_fases.grupo_espacial(elegida)
        if local.get("disponible"):
            salida["grupo_espacial"] = local
        return salida


def crear_selector_fase(config: dict, candidatos: Sequence[Any], *,
                        runtime=None,
                        project_root: Path | None = None) -> Callable | None:
    """El selector para una ronda, o None si aqui no se puede elegir fase.

    Devolver None no es un fallo: es el camino documentado de
    `RelaxationJobPreparer`, que entonces prepara la cubica de siempre. Elegir
    fase necesita un potencial interatomico y no todas las instalaciones lo
    tienen.
    """
    if not bool(_config_fases(config).get("activo", True)):
        log.info("seleccion de fase desactivada en config (discovery.fases.activo)")
        return None
    if not candidatos:
        return None

    if runtime is None:
        # Salir a buscar el MLFF de esta maquina es lo unico que puede tardar o
        # arrancar WSL. Con un runtime inyectado no hay nada que buscar, asi que
        # el interruptor solo aplica aqui --- es lo que permite que las pruebas
        # ejerciten la seleccion sin depender de como este montada la maquina.
        if os.environ.get("BUHO_FASES", "").strip().lower() in _APAGADO:
            log.info("BUHO_FASES apagado: el DFT se prepara sobre la fase cubica")
            return None
        try:
            runtime = resolve_mlff(config, project_root=project_root)
        except MLFFUnavailableError as exc:
            log.warning("sin runtime MLFF para elegir fase (%s); se prepara la cubica", exc)
            return None

    if getattr(runtime, "backend", "off") == "off":
        log.info("MLFF desactivado: el DFT se prepara sobre la fase cubica")
        return None

    # Deliberadamente NO se sondea el entorno antes. `--preflight-only` importa
    # torch y matgl, que es la parte cara; la llamada real vuelve a hacerlo y
    # ademas carga el potencial. Sondear duplicaba el coste para averiguar algo
    # que la propia llamada dice, y con peor mensaje: si el entorno no esta, el
    # error que se registra es el de la operacion que de verdad se queria hacer.
    return _SelectorPorLote(config, runtime, candidatos, project_root=project_root)
