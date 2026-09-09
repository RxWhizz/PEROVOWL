"""Cascada de cribado HTS expuesta por la API.

Envuelve `buho.screening.ScreeningCascade`: genera un lote reproducible de
candidatos y los pasa por los tres tiers, cada uno unas mil veces más caro que
el anterior (descriptores físicos → surrogate de bandgap → MLFF de estabilidad).

El cribado tarda minutos, así que corre en un hilo y se consulta por `run_id`.
Cada tier puede faltar —el `.pkl` del surrogate, o torch/matgl para el MLFF— y
en ese caso se declara ausente en vez de fallar: un ranking sin estabilidad
sigue siendo útil mientras se diga que le falta esa señal.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .. import paths

log = logging.getLogger(__name__)

# Cuántas ejecuciones se recuerdan.
MAX_RUNS = 20

# Reentrante: `_recortar_historial` y `_cargar_historial` se llaman tanto desde
# dentro como desde fuera de la sección crítica.
_lock = threading.RLock()
_runs: dict[str, ScreeningRun] = {}
_historial_cargado = False


# ── Configuración ────────────────────────────────────────────────────────────

def config_path() -> Path:
    """Config del generador: primero la del usuario, si no la empaquetada.

    Sin el fallback al bundle, el cribado en el binario congelado fallaba con
    `No se encuentra <data_root>/config/generator.yaml` en cuanto la raíz de
    datos no tenía una copia propia — que es el caso en cualquier instalación
    nueva. `services.discovery.config_path` ya hacía esto.
    """
    data_cfg = paths.resolve_data("config/generator.yaml")
    if data_cfg.is_file():
        return data_cfg
    return paths.bundle_file("config", "generator.yaml")


def load_generator_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _models_root() -> Path:
    """Raíz que `ScreeningCascade` usa para resolver `models/<modelo>.pkl`.

    La cascada compone `project_root / "models" / …`, así que se le pasa el
    padre del directorio de modelos que resuelva `paths` — empaquetado o del
    repositorio, según el modo.
    """
    return paths.find_resource("models").parent


# ── Disponibilidad de cada tier ──────────────────────────────────────────────

def _puede_importar(*modulos: str) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in modulos)


def tier_availability(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Qué tiers pueden ejecutarse aquí y por qué no, si no pueden."""
    cfg = cfg or {}
    scr = cfg.get("screening", {}) or {}

    surrogate_pkl = _models_root() / "models" / "surrogate_bandgap.pkl"
    tiene_surrogate = surrogate_pkl.is_file() and _puede_importar("sklearn")
    tiene_mlff = _puede_importar("torch", "matgl", "pymatgen")

    return [
        {
            "tier": 0,
            "name": "Física",
            "enabled": True,
            "available": True,
            "reason": None,
            "cost_hint": "~10 µs por candidato",
        },
        {
            "tier": 1,
            "name": "Surrogate",
            "enabled": bool(scr.get("tier1_surrogate", True)),
            "available": tiene_surrogate,
            "reason": None if tiene_surrogate else (
                f"falta {surrogate_pkl.name} o scikit-learn"
            ),
            "cost_hint": "~2 ms por candidato",
        },
        {
            "tier": 2,
            "name": "MLFF",
            "enabled": bool(scr.get("tier2_mlff", True)),
            "available": tiene_mlff,
            "reason": None if tiene_mlff else "faltan torch, matgl o pymatgen",
            "cost_hint": "~0.5 s por candidato",
        },
    ]


def gates(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cotas y umbrales que aplica la cascada, para dibujarlos en la interfaz."""
    filtros = cfg.get("filters", {}) or {}
    scr = cfg.get("screening", {}) or {}
    acq = cfg.get("acquisition", {}) or {}
    esp = cfg.get("chemical_space", {}) or {}

    return {
        "goldschmidt": filtros.get("goldschmidt", {"min": 0.80, "max": 1.10}),
        "octahedral": filtros.get("octahedral", {"min": 0.40, "max": 0.90}),
        "volume_A3": filtros.get("volume_A3", {"min": 50.0, "max": 2000.0}),
        "pv_window": acq.get("pv_window", [1.1, 1.8]),
        "eform_max_eV_atom": scr.get("eform_max_eV_atom", 0.20),
        "beta": acq.get("beta", 1.0),
        "n_dft_per_batch": scr.get("n_dft_per_batch", 0),
        "batch_size": (cfg.get("generation", {}) or {}).get("batch_size", 1000),
        "chemical_space": {
            "A": esp.get("A_sites", []),
            "B": esp.get("B_sites", []),
            "X": esp.get("X_sites", []),
        },
    }


# ── Ejecución ────────────────────────────────────────────────────────────────

@dataclass
class ScreeningRun:
    run_id: str
    batch_id: int
    n_requested: int
    use_mlff: bool
    random_seed: int = 42
    n_batches: int = 1
    n_candidates_per_batch: int | None = None
    lot_ids: list[int] = field(default_factory=list)
    status: str = "pending"          # pending | running | done | error
    stage: str = "cola"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    tiers: list[dict[str, Any]] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    n_selected: int = 0
    selected_candidate_ids: list[str] = field(default_factory=list)
    selected_candidates: list[dict[str, Any]] = field(default_factory=list, repr=False)
    dft_batch_path: str | None = None
    dft_prepared: int | None = None
    dft_started_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "n_requested": self.n_requested,
            "random_seed": self.random_seed,
            "n_batches": self.n_batches,
            "n_candidates_per_batch": self.n_candidates_per_batch or self.n_requested,
            "lot_ids": self.lot_ids,
            "use_mlff": self.use_mlff,
            "status": self.status,
            "stage": self.stage,
            "started_at": self.started_at,
            "elapsed_sec": round((self.finished_at or time.time()) - self.started_at, 1),
            "error": self.error,
            "tiers": self.tiers,
            "n_selected": self.n_selected,
            "selected_candidate_ids": self.selected_candidate_ids,
            "dft_batch_path": self.dft_batch_path,
            "dft_prepared": self.dft_prepared,
            "dft_started_at": self.dft_started_at,
            "items": self.items,
            "dropped": self.dropped,
        }


# ── Persistencia ─────────────────────────────────────────────────────────────
#
# El historial vivía solo en memoria: una criba de 300 candidatos devolvía sus
# seleccionados y se perdían al cerrar la app. Peor que rehacer el trabajo —que
# es barato— es que `start_dft_for_run` necesita `selected_candidates` enteros
# para materializar los jobs, así que sin esto un cribado no sobrevivía a un
# reinicio ni para lanzar el DFT que lo justificaba.

_CAMPOS_RUN = frozenset(f.name for f in fields(ScreeningRun))


def _dir_historial() -> Path:
    return paths.data_root() / "data" / "screening" / "runs"


def _guardar(run: ScreeningRun) -> None:
    """Vuelca una ejecución terminada. Nunca revienta al llamante."""
    if run.status in ("pending", "running"):
        return
    destino = _dir_historial() / f"{run.run_id}.json"
    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atómica: si la app muere a medias, el fichero anterior sigue
        # siendo válido en vez de quedar un JSON truncado que no se puede releer.
        parcial = destino.with_suffix(".json.parcial")
        parcial.write_text(
            json.dumps(asdict(run), ensure_ascii=False), encoding="utf-8"
        )
        parcial.replace(destino)
    except (OSError, TypeError, ValueError) as exc:
        log.warning("No se pudo persistir el cribado %s: %s", run.run_id, exc)


def _cargar_historial() -> None:
    """Relee las ejecuciones de disco la primera vez que se consulta el historial."""
    global _historial_cargado
    with _lock:
        if _historial_cargado:
            return
        _historial_cargado = True
        directorio = _dir_historial()
        if not directorio.is_dir():
            return
        for fichero in directorio.glob("*.json"):
            try:
                datos = json.loads(fichero.read_text(encoding="utf-8"))
                run = ScreeningRun(
                    **{k: v for k, v in datos.items() if k in _CAMPOS_RUN}
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                # Un fichero corrupto no puede impedir ver el resto del historial.
                log.warning("Cribado persistido ilegible en %s: %s", fichero, exc)
                continue
            _runs.setdefault(run.run_id, run)
        _recortar_historial()


def _recortar_historial() -> None:
    if len(_runs) <= MAX_RUNS:
        return
    viejos = sorted(_runs.values(), key=lambda r: r.started_at)[: len(_runs) - MAX_RUNS]
    for r in viejos:
        _runs.pop(r.run_id, None)
        try:
            (_dir_historial() / f"{r.run_id}.json").unlink(missing_ok=True)
        except OSError as exc:
            log.warning("No se pudo borrar el cribado %s: %s", r.run_id, exc)


def start_run(
    *,
    batch_id: int | None = None,
    n_candidates: int,
    n_batches: int = 1,
    random_seed: int | None = None,
    use_mlff: bool | None = None,
) -> ScreeningRun:
    """Lanza la cascada en segundo plano y devuelve el registro de la ejecución."""
    _cargar_historial()
    cfg = load_generator_config()
    cfg_seed = int(cfg.get("random_seed", 42))
    effective_seed = cfg_seed if random_seed is None else int(random_seed)

    # Compatibilidad: la API vieja usaba `batch_id` como offset de semilla.
    legacy_single_batch = random_seed is None and n_batches == 1 and batch_id is not None
    lot_ids = [int(batch_id)] if legacy_single_batch else list(range(int(n_batches)))
    output_batch_id = int(batch_id) if batch_id is not None else effective_seed

    disponibles = {t["tier"]: t for t in tier_availability(cfg)}
    if use_mlff is None:
        use_mlff = bool((cfg.get("screening", {}) or {}).get("tier2_mlff", True))
    # Pedir el Tier 2 sin sus dependencias no es un error: se desactiva y se dice.
    if use_mlff and not disponibles[2]["available"]:
        log.warning("Tier 2 pedido pero no disponible: %s", disponibles[2]["reason"])
        use_mlff = False

    run = ScreeningRun(
        run_id=uuid.uuid4().hex[:12],
        batch_id=output_batch_id,
        n_requested=n_candidates * len(lot_ids),
        n_candidates_per_batch=n_candidates,
        n_batches=len(lot_ids),
        random_seed=effective_seed,
        lot_ids=lot_ids,
        use_mlff=use_mlff,
    )
    with _lock:
        _runs[run.run_id] = run
        _recortar_historial()

    threading.Thread(target=_ejecutar, args=(run, cfg), daemon=True).start()
    return run


def get_run(run_id: str) -> ScreeningRun | None:
    _cargar_historial()
    return _runs.get(run_id)


def list_runs() -> list[dict[str, Any]]:
    """Historial reciente, sin las filas de resultados (que son grandes)."""
    _cargar_historial()
    resumen = []
    for r in sorted(_runs.values(), key=lambda x: x.started_at, reverse=True):
        d = r.as_dict()
        d.pop("items", None)
        d.pop("dropped", None)
        resumen.append(d)
    return resumen


def start_dft_for_run(poller, run_id: str, *, start_runner: bool = True) -> dict[str, Any]:
    """Materializa los seleccionados del cribado como jobs DFT y opcionalmente arranca el runner."""
    run = get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    if run.status != "done":
        raise RuntimeError(f"El cribado esta en estado '{run.status}', no 'done'.")
    if not run.selected_candidates:
        raise RuntimeError("El cribado no tiene candidatos seleccionados para DFT.")

    from buho.dft_jobs.prepare_relaxation_jobs import RelaxationJobPreparer
    from buho.generator.heuristic_generator import GeneratedCandidate

    from .control import ControlError, start_batch
    from .. import platform_caps

    cfg = load_generator_config()
    batch_root = _screening_batch_root(poller)
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_dir = batch_root / f"batch_{run.batch_id:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    candidates = [GeneratedCandidate.from_dict(item) for item in run.selected_candidates]
    python_exe = platform_caps.runner_python(poller.cfg) or "python3"
    # Misma decision que en el protocolo autonomo: la fase se mide si hay
    # potencial, y si no se prepara la cubica de siempre. El boton de la GUI no
    # puede quedarse con otro criterio que el bucle.
    from buho.structure.selector_mlff import crear_selector_fase

    selector = crear_selector_fase(cfg, candidates, project_root=paths.data_root())
    preparer = RelaxationJobPreparer(
        cfg,
        project_root=paths.data_root(),
        n_cores=int(poller.cfg.get("runner_cores", 8)),
        python=python_exe,
        selector_fase=selector,
    )
    prepared = preparer.prepare(candidates, out_root=batch_dir, config_src=config_path())
    _mark_screening_passed_jobs(batch_dir, run, candidates)

    launched = False
    runner_error: str | None = None
    runner_kind = str(poller.cfg.get("runner_kind", "relax"))
    if start_runner:
        if not platform_caps.runner_launch_available(poller.cfg):
            runner_error = (
                "No hay interprete Python o scripts del pipeline disponibles para lanzar DFT."
            )
        else:
            try:
                start_batch(poller, run.batch_id)
                launched = True
                run.dft_started_at = time.time()
            except ControlError as exc:
                runner_error = str(exc)

    run.dft_batch_path = str(batch_dir)
    run.dft_prepared = len(prepared)
    # Sin esto, reabrir la app perdería el vínculo entre el cribado y el lote
    # DFT que salió de él.
    _guardar(run)

    return {
        "run_id": run.run_id,
        "batch_id": run.batch_id,
        "batch_path": str(batch_dir),
        "n_selected": len(candidates),
        "n_prepared": len(prepared),
        "n_existing_or_skipped": max(0, len(candidates) - len(prepared)),
        "runner_launched": launched,
        "runner_kind": runner_kind,
        "runner_error": runner_error,
    }


def _mark_screening_passed_jobs(batch_dir: Path, run: ScreeningRun, candidates: list[Any]) -> None:
    """Marca jobs materializados desde la selección final del cribado."""
    selected_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "candidate_id", ""))
        if not candidate_id:
            continue

        job_dir = batch_dir / candidate_id
        if not job_dir.is_dir():
            continue

        metadata_path = job_dir / "metadata.json"
        data: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}

        if hasattr(candidate, "to_dict"):
            candidate_data = candidate.to_dict()
            for key in (
                "candidate_id",
                "formula",
                "generation_mode",
                "A_site_species",
                "B_site_species",
                "X_site_species",
                "fractions",
            ):
                if key in candidate_data and key not in data:
                    data[key] = candidate_data[key]

        data.update({
            "screening_passed_tiers": True,
            "screening_run_id": run.run_id,
            "screening_random_seed": run.random_seed,
            "screening_lot_ids": run.lot_ids,
            "screening_selected_candidate_ids": run.selected_candidate_ids,
            "screening_selected_at": selected_at,
        })
        metadata_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _screening_batch_root(poller) -> Path:
    """Usa la misma raiz de batches que el control del monitor."""
    from .control import _raiz_batches

    return _raiz_batches(poller)


def reset_runs() -> None:
    """Olvida el historial en memoria (para tests). No toca lo persistido."""
    global _historial_cargado
    with _lock:
        _runs.clear()
        _historial_cargado = False


def _ejecutar(run: ScreeningRun, cfg: dict[str, Any]) -> None:
    try:
        run.status = "running"
        t0 = time.time()

        # ── Generar ──────────────────────────────────────────────────────────
        run.stage = "generando"
        from buho.generator.heuristic_generator import HeuristicGenerator

        gen = HeuristicGenerator(config_path(), random_seed=run.random_seed)
        candidatos = _generar_lotes(gen, run)
        t_gen = time.time() - t0
        n_generados = len(candidatos)

        if not candidatos:
            run.tiers = []
            run.status = "done"
            run.finished_at = time.time()
            return

        # ── Cascada ──────────────────────────────────────────────────────────
        run.stage = "tier 0 · física"
        from buho.screening.cascade import ScreeningCascade

        # `extra_roots`: el surrogate reentrenado por el protocolo autónomo vive
        # en la raíz de datos, no junto a los de fábrica. Sin esto, el cribado
        # manual seguiría usando el modelo original aunque el bucle ya hubiera
        # aprendido.
        cascada = ScreeningCascade(cfg, project_root=_models_root(),
                                   extra_roots=(paths.data_root(),))
        t1 = time.time()
        run.stage = "tier 1 · surrogate" if not run.use_mlff else "tier 1-2 · ML"
        df = cascada.screen(candidatos, run_mlff=run.use_mlff)
        t_screen = time.time() - t1

        run.stage = "seleccionando"
        seleccion = cascada.select_for_dft(df)
        selected_ids = []
        if "candidate_id" in seleccion:
            selected_ids = [
                str(cid) for cid in seleccion["candidate_id"].tolist()
                if cid is not None
            ]
        by_id = {c.candidate_id: c for c in candidatos}

        run.tiers = _resumen_tiers(df, n_generados, t_gen, t_screen, run.use_mlff, cfg)
        run.tiers.append({
            "tier": 3,
            "name": "Selección",
            "kind": "select",
            "n_in": int(run.tiers[-1]["n_out"]),
            "n_out": int(len(seleccion)),
            "n_dropped": max(0, int(run.tiers[-1]["n_out"]) - int(len(seleccion))),
            "note": "estables, ordenados por total_score",
            "seconds": None,
            "ran": True,
        })
        run.items = _filas(df)
        run.dropped = _descartes(df)
        run.n_selected = int(len(seleccion))
        run.selected_candidate_ids = selected_ids
        run.selected_candidates = [
            by_id[cid].to_dict() for cid in selected_ids if cid in by_id
        ]
        run.status = "done"
        run.stage = "listo"
    except Exception as exc:  # el hilo no debe morir en silencio
        log.exception("Cribado %s falló", run.run_id)
        run.status = "error"
        run.stage = "error"
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        run.finished_at = time.time()
        _guardar(run)


def _generar_lotes(gen, run: ScreeningRun):
    """Genera varios lotes reproducibles y elimina duplicados entre ellos."""
    por_id = {}
    for lot_id in run.lot_ids:
        lote = gen.generate_batch(
            lot_id,
            batch_size=run.n_candidates_per_batch or run.n_requested,
        )
        for candidato in lote:
            por_id.setdefault(candidato.candidate_id, candidato)
    return list(por_id.values())


def _descartes(df, limite: int = 60) -> list[dict[str, Any]]:
    """Muestra de descartados con su motivo, para poder auditar la torre."""
    if "dropped_at_tier" not in df:
        return []
    fuera = df[df["dropped_at_tier"].notna()]
    muestra = []
    for tier in sorted(fuera["dropped_at_tier"].unique()):
        trozo = fuera[fuera["dropped_at_tier"] == tier].head(limite // 3)
        for registro in trozo.to_dict(orient="records"):
            muestra.append({
                "formula": registro.get("formula"),
                "dropped_at_tier": int(tier),
                "drop_reason": registro.get("drop_reason"),
            })
    return muestra[:limite]


def _resumen_tiers(df, n_generados: int, t_gen: float, t_screen: float,
                   use_mlff: bool, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Embudo por tier, derivado de `dropped_at_tier`.

    `kind` sale de la configuración, no de una suposición: con `tier1_gate` a
    false ese tier vuelve a solo puntuar, y la interfaz debe decirlo en vez de
    dibujar una criba que no ocurre.
    """
    scr = cfg.get("screening", {}) or {}
    puertas = {
        1: bool(scr.get("tier1_gate", True)),
        2: bool(scr.get("tier2_gate", True)) and use_mlff,
    }
    sigma_k = scr.get("sigma_k", 1.0)

    def caidos(tier: int) -> int:
        if "dropped_at_tier" not in df:
            return 0
        return int((df["dropped_at_tier"] == tier).sum())

    total = int(len(df))
    vivos = total
    tiers: list[dict[str, Any]] = []

    definiciones = [
        (0, "Física", True, "descarta fuera de cotas", round(t_gen, 2)),
        (1, "Surrogate", puertas[1],
         f"ventana PV con {sigma_k}σ de holgura" if puertas[1]
         else "marca la ventana PV y aporta el UCB — no descarta",
         round(t_screen, 2)),
        (2, "MLFF", puertas[2],
         f"energía de formación con {sigma_k}σ de holgura" if puertas[2]
         else ("estima estabilidad — no descarta" if use_mlff else "no ejecutado"),
         None),
    ]

    for tier, nombre, es_puerta, nota, segundos in definiciones:
        n_in = vivos
        n_out = vivos - caidos(tier)
        tiers.append({
            "tier": tier,
            "name": nombre,
            "kind": "gate" if es_puerta else "signal",
            "n_in": n_in,
            "n_out": n_out,
            "n_dropped": n_in - n_out,
            "note": nota,
            "seconds": segundos,
            "ran": tier == 0 or (tier == 1) or use_mlff,
        })
        vivos = n_out

    return tiers


_COLUMNAS = (
    "candidate_id", "formula", "generation_mode", "tolerance_t", "oct_factor",
    "vol_est_A3", "Eg_surrogate_eV", "Eg_sigma_eV", "band_score", "in_pv_window",
    "Eform_eV_atom", "is_stable", "stab_score", "ucb_bonus", "total_score",
    "passed_eform", "tier_reached", "dropped_at_tier", "drop_reason",
)


def _filas(df, limite: int = 500, *, solo_vivos: bool = True) -> list[dict[str, Any]]:
    """Filas saneadas para JSON (el NaN de pandas no es serializable)."""
    import math

    if solo_vivos and "dropped_at_tier" in df:
        df = df[df["dropped_at_tier"].isna()]
    if "total_score" in df:
        df = df.sort_values("total_score", ascending=False)

    filas = []
    for registro in df.head(limite).to_dict(orient="records"):
        fila = {}
        for clave in _COLUMNAS:
            valor = registro.get(clave)
            if isinstance(valor, float) and not math.isfinite(valor):
                valor = None
            elif hasattr(valor, "item"):
                valor = valor.item()
            fila[clave] = valor
        filas.append(fila)
    return filas
