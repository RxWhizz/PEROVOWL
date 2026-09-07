"""Endpoints REST del monitor DFT. El WebSocket vive en `ws.py`."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from . import __version__, paths, platform_caps
from .models import (
    JobStatus,
    PingResponse,
    StatsResponse,
    SummaryResponse,
)
from .poller import DFTPoller, ping_job
from .sysmetrics import SysMetrics
from .sysmetrics import collect as collect_metrics
from .sysmetrics import format_telegram as fmt_sys
from .utils import fmt_formula

log = logging.getLogger(__name__)

router = APIRouter()


def get_poller(request: Request) -> DFTPoller:
    return request.app.state.poller


# ── REST endpoints ────────────────────────────────────────────────────────────

class PathsInfo(BaseModel):
    frozen: bool
    bundle_root: str
    data_root: str
    config_dir: str


class PlatformInfo(BaseModel):
    os: str
    frozen: bool
    hardware_temps: bool
    runner_launch: bool
    runner_python: str | None = None
    auto_advance: bool = True


class HealthResponse(BaseModel):
    ok: bool
    version: str
    paths: PathsInfo
    platform: PlatformInfo
    runs_dir: str
    runs_mounted: bool
    runs_state: str
    nearest_existing_path: str
    n_jobs_tracked: int
    last_poll_at: float | None
    last_poll_age_sec: float | None
    poll_interval_sec: int
    ws_clients: int


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Salud del monitor: montaje de `runs_dir`, frescura del poller, clientes WS.

    `runs_mounted` existe porque `runs/` y `calculations/` son symlinks a un
    volumen externo. Sin esta señal, "el disco está desmontado" y "no hay jobs"
    se ven exactamente igual desde el cliente.

    `runs_state` desglosa ese booleano en tres, porque "el directorio no está"
    tampoco distingue por sí solo un volumen caído de una instalación recién
    hecha donde todavía no se ha lanzado nada. Al usuario que acaba de instalar
    se le decía DESMONTADO y salud "Atención" sin que hubiera nada roto.
    """
    poller = get_poller(request)
    runs_dir = poller.runs_dir

    # Ancestro más cercano que sí existe: señala dónde se rompe la cadena.
    nearest = runs_dir
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent

    last_poll = getattr(poller, "last_poll_at", None)
    age = (time.time() - last_poll) if last_poll else None
    interval = int(poller.cfg.get("poll_interval_sec", 30))
    hub = getattr(request.app.state, "hub", None)

    mounted = runs_dir.is_dir()
    fresh = age is not None and age < 3 * interval

    # Lo que separa "volumen caido" de "instalacion nueva" es de donde cuelga
    # runs_dir: dentro de la raiz de datos es un directorio que el pipeline aun
    # no ha creado; fuera, es el symlink al disco externo y su ausencia si es un
    # fallo. Un enlace que no resuelve, o jobs que teniamos y ya no vemos, lo son
    # en cualquier caso.
    try:
        dentro_de_datos = runs_dir.is_relative_to(paths.data_root())
    except (OSError, ValueError):
        dentro_de_datos = False
    enlace_roto = runs_dir.is_symlink() and not runs_dir.exists()

    if mounted:
        runs_state = "montado"
    elif enlace_roto or poller.snapshots or not dentro_de_datos:
        runs_state = "desmontado"
    else:
        runs_state = "sin_estrenar"

    return HealthResponse(
        ok=fresh and runs_state != "desmontado",
        version=__version__,
        # Dónde busca cada cosa: sin esto, "no veo mis jobs" no es diagnosticable
        # en un binario, donde la raíz de datos ya no es el repositorio.
        paths=PathsInfo(**paths.describe()),
        # Capacidades comprobadas, no supuestas por sistema operativo: el
        # frontend esconde lo que no está disponible en vez de ofrecer acciones
        # que van a fallar.
        platform=PlatformInfo(**platform_caps.describe(poller.cfg)),
        runs_dir=str(runs_dir),
        runs_mounted=mounted,
        runs_state=runs_state,
        nearest_existing_path=str(nearest),
        n_jobs_tracked=len(poller.snapshots),
        last_poll_at=last_poll,
        last_poll_age_sec=round(age, 1) if age is not None else None,
        poll_interval_sec=interval,
        ws_clients=hub.n_clients if hub else 0,
    )


def _to_job_status(s: StatsResponse) -> JobStatus:
    return JobStatus(
        job_id=s.job_id,
        formula=s.formula,
        status=s.status,
        pid=s.pid,
        start_time=s.start_time,
        elapsed_min=s.elapsed_min,
        mpi_cores=s.mpi_cores,
    )


class JobPage(BaseModel):
    items: list[JobStatus]
    total: int
    limit: int
    offset: int


_SORT_KEYS = {
    "formula":     lambda s: (s.formula or "").lower(),
    "job_id":      lambda s: s.job_id,
    "status":      lambda s: s.status,
    "elapsed_min": lambda s: s.elapsed_min or 0.0,
}


@router.get("/api/jobs", response_model=JobPage)
async def list_jobs(
    request: Request,
    status: str | None = Query(None, description="Estado a filtrar; admite lista separada por comas."),
    q: str | None = Query(None, description="Subcadena buscada en la fórmula o el job_id."),
    sort: str = Query("formula", pattern="^(formula|job_id|status|elapsed_min)$"),
    desc: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> JobPage:
    """Jobs paginados, con filtro y orden.

    Devolver los ~2500 snapshots enteros en cada carga eran varios MB de JSON.
    """
    poller = get_poller(request)
    snaps = list(poller.snapshots.values())

    if status:
        wanted = {part.strip() for part in status.split(",") if part.strip()}
        snaps = [s for s in snaps if s.status in wanted]

    if q:
        needle = q.lower()
        snaps = [
            s for s in snaps
            if needle in (s.formula or "").lower() or needle in s.job_id.lower()
        ]

    snaps.sort(key=_SORT_KEYS[sort], reverse=desc)

    return JobPage(
        items=[_to_job_status(s) for s in snaps[offset:offset + limit]],
        total=len(snaps),
        limit=limit,
        offset=offset,
    )


# `/api/jobs/converged` DEBE declararse antes que `/api/jobs/{job_id}`: Starlette
# resuelve por orden de registro, así que al revés la ruta paramétrica la captura
# con job_id="converged" y devuelve 404 siempre.
@router.get("/api/jobs/converged", response_model=list[JobStatus])
async def list_converged(request: Request, limit: int = 50) -> list[JobStatus]:
    """Lista los primeros N jobs convergidos ordenados por fórmula."""
    poller = get_poller(request)
    jobs = [
        _to_job_status(s)
        for s in poller.snapshots.values()
        if s.status == "converged"
    ]
    jobs.sort(key=lambda j: j.formula)
    return jobs[:limit]


@router.get("/api/jobs/{job_id}", response_model=StatsResponse)
async def get_job(job_id: str, request: Request) -> StatsResponse:
    poller = get_poller(request)
    snap = poller.snapshots.get(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado")
    return snap


@router.get("/api/jobs/{job_id}/ping", response_model=PingResponse)
async def ping(job_id: str, request: Request) -> PingResponse:
    """Lectura instantánea del log actual — no usa caché."""
    poller   = get_poller(request)
    job_dir  = poller.runs_dir / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado")

    data = ping_job(job_dir)
    return PingResponse(job_id=job_id, **data)


@router.get("/api/jobs/{job_id}/stats", response_model=StatsResponse)
async def get_stats(job_id: str, request: Request) -> StatsResponse:
    """Igual que GET /api/jobs/{job_id} pero fuerza re-parseo del disco."""
    from .poller import snapshot_job
    poller  = get_poller(request)
    job_dir = poller.runs_dir / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado")
    snap = snapshot_job(job_dir, poller.cfg)
    poller.snapshots[job_id] = snap
    return snap


# ── Artefactos de un job ─────────────────────────────────────────────────────

def _job_dir_or_404(request: Request, job_id: str) -> Path:
    """Resuelve el directorio del job validando el id contra path traversal."""
    from .services.jobs import UnsafeJobIdError, resolve_job_dir

    poller = get_poller(request)
    try:
        job_dir = resolve_job_dir(poller.runs_dir, job_id)
    except UnsafeJobIdError:
        raise HTTPException(status_code=400, detail="job_id inválido") from None
    if job_dir is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado")
    return job_dir


@router.get("/api/jobs/{job_id}/log")
async def job_log(
    request: Request,
    job_id: str,
    label: str | None = Query(None, description="Sub-cálculo; por defecto el primero disponible."),
    tail: int = Query(200, ge=1, le=2000, description="Últimas N líneas."),
) -> dict:
    """Cola del log del job. `available` lista las etiquetas seleccionables."""
    from .services.jobs import read_log_tail

    return read_log_tail(_job_dir_or_404(request, job_id), label, tail)


@router.get("/api/jobs/{job_id}/traces")
async def job_traces_endpoint(request: Request, job_id: str) -> dict:
    """Series SCF por etiqueta y resumen de los frames etiquetados con DFT."""
    from .services.jobs import job_traces

    return job_traces(_job_dir_or_404(request, job_id))


@router.get("/api/jobs/{job_id}/metadata")
async def job_metadata_endpoint(request: Request, job_id: str) -> dict:
    """metadata.json y status.json del job, más el inventario de artefactos."""
    from .services.jobs import job_metadata

    return job_metadata(_job_dir_or_404(request, job_id))


@router.get("/api/summary", response_model=SummaryResponse)
async def summary(request: Request) -> SummaryResponse:
    poller = get_poller(request)
    snaps  = list(poller.snapshots.values())
    counts: dict[str, int] = {}
    for s in snaps:
        counts[s.status] = counts.get(s.status, 0) + 1

    n_conv = counts.get("converged", 0)
    n_fail = counts.get("failed", 0)
    rate   = round(n_conv / (n_conv + n_fail), 3) if (n_conv + n_fail) > 0 else None

    return SummaryResponse(
        n_pending=counts.get("pending", 0),
        n_running=counts.get("running", 0),
        n_converged=n_conv,
        n_failed=n_fail,
        n_stalled=counts.get("stalled", 0),
        n_oscillating=counts.get("oscillating", 0),
        n_skipped_duplicate=counts.get("skipped_duplicate", 0),
        total=len(snaps),
        convergence_rate=rate,
    )


class SysMetricsResponse(BaseModel):
    cpu_percent: float
    cpu_per_core: list[float]
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    pkg_temps: list[float]
    core_temp_max: float
    nvme_temp: float | None
    gpu_temps: list[float]


@router.get("/api/system", response_model=SysMetricsResponse)
async def system_metrics() -> SysMetricsResponse:
    """Temperaturas, uso de CPU y RAM en tiempo real."""
    m = collect_metrics()
    return SysMetricsResponse(**m.__dict__)


class MetricsHistoryResponse(BaseModel):
    samples: list[dict]
    interval_sec: int


@router.get("/api/system/history", response_model=MetricsHistoryResponse)
async def system_history(
    request: Request,
    minutes: int = Query(10, ge=1, le=60, description="Ventana en minutos."),
) -> MetricsHistoryResponse:
    """Serie reciente de CPU, RAM y temperatura, para las sparklines.

    `/api/system` solo da el instante actual.
    """
    history = getattr(request.app.state, "metrics_history", None)
    if history is None:
        return MetricsHistoryResponse(samples=[], interval_sec=0)
    from .metrics_history import DEFAULT_INTERVAL_SEC
    return MetricsHistoryResponse(
        samples=history.samples(since_sec=minutes * 60),
        interval_sec=DEFAULT_INTERVAL_SEC,
    )


def _count_total_converged(poller: DFTPoller) -> int:
    """Cuenta convergidos en todos los batches históricos + relax_basic."""
    total = 0

    def cfg_path(key: str, default: str) -> Path:
        return paths.resolve_data(poller.cfg.get(key, default))

    for search_dir in [
        cfg_path("runs_dir", "runs/relax_basic"),
        cfg_path("batches_dir", "runs/batches"),
    ]:
        if not search_dir.exists():
            continue
        # relax_basic: hijos directos son job_dirs
        # batches: hijos son batch_NNN/, nietos son job_dirs
        depth1 = [d for d in search_dir.iterdir() if d.is_dir()]
        for d1 in depth1:
            s = d1 / "status.json"
            if s.exists():
                # job directo (relax_basic)
                try:
                    if json.loads(s.read_text()).get("status") == "converged":
                        total += 1
                except Exception:
                    pass
            else:
                # batch_NNN — iterar nietos
                for d2 in d1.iterdir():
                    if not d2.is_dir():
                        continue
                    s2 = d2 / "status.json"
                    try:
                        if s2.exists() and json.loads(s2.read_text()).get("status") == "converged":
                            total += 1
                    except Exception:
                        pass
    return total


def _build_status_report(poller: DFTPoller, metrics: SysMetrics) -> str:
    """Genera el texto HTML del reporte tipo STATUS para Telegram."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    snaps = list(poller.snapshots.values())

    counts: dict[str, int] = {}
    for s in snaps:
        counts[s.status] = counts.get(s.status, 0) + 1

    n_run  = counts.get("running",  0) + counts.get("stalled", 0) + counts.get("oscillating", 0)
    n_conv = counts.get("converged", 0)
    n_fail = counts.get("failed",   0) + counts.get("stopped", 0)
    n_pend = counts.get("pending",  0)

    batch_name = poller.runs_dir.name
    total_conv = _count_total_converged(poller)

    lines = [
        f"📊 <b>DFT Status</b> — {now}",
        f"📦 Batch: <code>{batch_name}</code>   🧮 Total simuladas: <b>{total_conv}</b>",
        "",
        fmt_sys(metrics),
        "",
        f"📋 <b>Jobs</b>: {n_run} corriendo | {n_conv} ✅ | {n_fail} ❌ | {n_pend} en cola",
    ]

    _ICON = {
        "running":     "🔄",
        "converged":   "✅",
        "failed":      "❌",
        "stopped":     "🔴",
        "stalled":     "⏸️",
        "oscillating": "⚠️",
        "pending":     "⏳",
        "skipped_duplicate": "♻️",
        "unknown":     "❓",
    }

    # Activos: detalle completo
    active = [s for s in snaps if s.status in ("running", "oscillating", "stalled")]
    for s in sorted(active, key=lambda s: s.job_id):
        icon = _ICON.get(s.status, "❓")
        name = f"<code>{fmt_formula(s.formula or s.job_id)}</code>"
        parts = []
        if s.n_fire_steps:
            parts.append(f"FIRE {s.n_fire_steps}")
            if s.fmax_history:
                parts.append(f"fmax={s.fmax_history[-1]:.3f}")
        elif s.n_scf_iters:
            parts.append(f"SCF {s.n_scf_iters}")
        if s.energy_history:
            parts.append(f"E={s.energy_history[-1]:.3f} eV")
        if s.is_oscillating:
            parts.append("⚠️osc")
        if s.stall_minutes:
            parts.append(f"stall {s.stall_minutes:.0f}min")
        elapsed = f"  {s.elapsed_min:.0f}min" if s.elapsed_min else ""
        lines.append(f"{icon} {name}  {'  '.join(parts) or 'init'}{elapsed}")

    # Fallidos/detenidos: todos (suelen ser pocos)
    failed = [s for s in snaps if s.status in ("failed", "stopped")]
    for s in sorted(failed, key=lambda s: s.job_id):
        icon = _ICON.get(s.status, "❓")
        name = f"<code>{fmt_formula(s.formula or s.job_id)}</code>"
        t_str = f"  {s.elapsed_min:.0f}min" if s.elapsed_min else ""
        lines.append(f"{icon} {name}{t_str}")

    # Convergidos: solo los últimos 5
    converged = sorted(
        [s for s in snaps if s.status == "converged"],
        key=lambda s: s.elapsed_min or 0,
        reverse=True,
    )
    for s in converged[:5]:
        icon = _ICON["converged"]
        name = f"<code>{fmt_formula(s.formula or s.job_id)}</code>"
        e_str = f"  E={s.final_energy_ev:.3f} eV" if s.final_energy_ev else ""
        lines.append(f"{icon} {name}{e_str}")
    if len(converged) > 5:
        lines.append(f"  … +{len(converged)-5} más convergidos")

    return "\n".join(lines)


def _build_converged_text(poller: DFTPoller, limit: int = 50) -> str:
    """Construye el texto HTML del listado de convergidos (reutilizable por el bot)."""
    converged = sorted(
        [s for s in poller.snapshots.values() if s.status == "converged"],
        key=lambda s: s.formula,
    )
    MAX_CHARS = 4000
    header = f"✅ <b>Convergidos</b> ({len(converged)} total)\n\n"
    lines: list[str] = []
    for s in converged[:limit]:
        e_str = f"  E={s.final_energy_ev:.3f} eV" if s.final_energy_ev else ""
        line = f"• <code>{fmt_formula(s.formula)}</code>{e_str}"
        if len(header + "\n".join(lines + [line])) > MAX_CHARS:
            lines.append(f"…(truncado, mostrados {len(lines)} de {len(converged)})")
            break
        lines.append(line)
    return header + "\n".join(lines)


def _scf_rate_s(points: list[dict]) -> float | None:
    """Segundos por iteración. Implementación única en services.jobs."""
    from .services.jobs import _scf_rate_s as _impl

    return _impl(points)


_PHASE2_N_CONFIGS = 4
_SCF_TYPICAL_ITERS = 15


def _batch_eta(runs_dir: Path, n_recent: int = 8) -> tuple[float | None, int, int]:
    """Throughput-based ETA for the current batch.

    Returns (rate_per_hour, n_pending, n_running).
    Uses pbe/label.extxyz mtime as convergence timestamp (reliable proxy).
    """
    conv_times: list[float] = []
    n_pending = n_running = 0
    for st_path in runs_dir.glob("*/status.json"):
        try:
            d = json.loads(st_path.read_text())
        except Exception:
            continue
        s = d.get("status")
        if s == "converged":
            xyz = st_path.parent / "pbe" / "label.extxyz"
            if xyz.exists():
                conv_times.append(xyz.stat().st_mtime)
        elif s == "pending":
            n_pending += 1
        elif s == "running":
            n_running += 1

    if len(conv_times) < 2:
        return None, n_pending, n_running

    conv_times.sort()
    recent = conv_times[-n_recent:]
    window_h = (recent[-1] - recent[0]) / 3600.0
    rate = (len(recent) - 1) / window_h if window_h > 0 else None
    return rate, n_pending, n_running


def _build_statusfull_report(poller: DFTPoller, metrics: SysMetrics) -> list[str]:
    """Per-SCF-iteration detail for running Phase 2A jobs (/statusfull command)."""
    from buho.phase2_force.self_heal import parse_scf_points

    now = datetime.now().strftime("%H:%M")
    batch_name = poller.runs_dir.name
    active = [
        s for s in poller.snapshots.values()
        if s.status in ("running", "stalled", "oscillating")
    ]

    if not active:
        counts: dict[str, int] = {}
        for s in poller.snapshots.values():
            counts[s.status] = counts.get(s.status, 0) + 1
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        rate, n_pend, _ = _batch_eta(poller.runs_dir)
        if rate and n_pend > 0:
            eta_h = n_pend / rate
            eta_str = f"ETA ~{eta_h*60:.0f}min" if eta_h < 1 else f"ETA ~{eta_h:.1f}h"
            rate_line = f"📈 {rate:.1f} jobs/h · {eta_str} ({n_pend} pendientes)"
        else:
            rate_line = f"📈 {n_pend} pendientes (runner pausado)"
        return [
            f"🔬 <b>Status full</b> — {now} | <code>{batch_name}</code>\n"
            f"Sin jobs activos  ({summary})\n"
            f"{rate_line}\n"
            f"{fmt_sys(metrics)}"
        ]

    ram_free = metrics.ram_total_gb - metrics.ram_used_gb
    rate, n_pend, n_run = _batch_eta(poller.runs_dir)
    n_conv = sum(1 for s in poller.snapshots.values() if s.status == "converged")
    n_total = len(poller.snapshots)
    if rate and (n_pend + n_run) > 0:
        eta_h = (n_pend + n_run) / rate
        if eta_h < 1:
            eta_str = f"ETA ~{eta_h*60:.0f}min"
        else:
            eta_str = f"ETA ~{eta_h:.1f}h"
        rate_str = f"{rate:.1f} jobs/h · {eta_str}"
    else:
        rate_str = "calculando tasa…"
    header = (
        f"🔬 <b>Status full</b> — {now} | <code>{batch_name}</code> | "
        f"{len(active)} activos | {n_conv}/{n_total} conv\n"
        f"📈 {rate_str} | RAM {ram_free:.0f} GB libre\n"
        f"{fmt_sys(metrics)}"
    )

    MAX_MSG = 4000
    messages: list[str] = []
    current_parts: list[str] = [header]
    current_len = len(header)

    for s in sorted(active, key=lambda x: x.job_id):
        job_dir = poller.runs_dir / s.job_id
        pbe_dir = job_dir / "pbe"

        n_done = 0
        frame_energies: list[str] = []
        for k in range(_PHASE2_N_CONFIGS):
            fj = pbe_dir / f"frame_{k}.json"
            if fj.exists():
                try:
                    fdata = json.loads(fj.read_text())
                    if fdata.get("status") == "ok":
                        e = fdata.get("energy_ev")
                        e_str = f"E={e:.4f}" if e is not None else "ok"
                        frame_energies.append(f"cfg{k}:{e_str}")
                        n_done += 1
                except Exception:
                    pass

        current_config = n_done
        config_label = f"config {current_config + 1}/{_PHASE2_N_CONFIGS}"

        log_path = pbe_dir / "r2scan.txt"
        points: list[dict] = []
        if log_path.exists():
            try:
                points = parse_scf_points(log_path)
            except Exception:
                pass

        formula = s.formula or s.job_id[:8]
        job_id_short = s.job_id[:8]
        elapsed = f"{s.elapsed_min:.1f} min" if s.elapsed_min else "?"

        job_lines = [
            f"\n🚀 <b>{fmt_formula(formula)}</b> (<code>{job_id_short}</code>) — "
            f"{config_label} · {elapsed}"
        ]

        if frame_energies:
            job_lines.append("  frames: " + "  ".join(frame_energies))

        if points:
            skip = len(points) - 12
            if skip > 0:
                job_lines.append(f"  … ({skip} iters previas)")
            for pt in points[-12:]:
                it = pt.get("iter", "?")
                clock = pt.get("clock", "?")
                e = pt.get("energy")
                e_str = f"E={e:.4f}" if e is not None else ""
                dens = pt.get("dens", "")
                job_lines.append(f"  iter {it:>3}  {clock}  {e_str}  dens={dens}")

            rate = _scf_rate_s(points)
            if rate is not None:
                last_iter = points[-1].get("iter") or 0
                remaining = max(0, _SCF_TYPICAL_ITERS - last_iter)
                eta_s = int(rate * remaining)
                eta_str = f"{eta_s // 60}min {eta_s % 60}s" if eta_s >= 60 else f"{eta_s}s"
                job_lines.append(f"  ritmo {rate:.0f} s/iter · ETA config ~{eta_str}")
        else:
            job_lines.append("  (sin iters SCF aún — inicializando)")

        block = "\n".join(job_lines)
        if current_len + len(block) > MAX_MSG:
            messages.append("\n".join(current_parts))
            current_parts = [block]
            current_len = len(block)
        else:
            current_parts.append(block)
            current_len += len(block)

    if current_parts:
        messages.append("\n".join(current_parts))

    return messages


@router.get("/api/statusfull")
async def statusfull_report(request: Request) -> dict:
    """Devuelve el reporte statusfull (detalle SCF por job activo) como lista de mensajes."""
    poller  = get_poller(request)
    metrics = collect_metrics()
    msgs    = _build_statusfull_report(poller, metrics)
    return {"messages": msgs, "count": len(msgs), "timestamp": datetime.now().isoformat()}


@router.post("/api/notify/converged")
async def notify_converged(request: Request, limit: int = 50) -> dict:
    """Envía a Telegram los primeros N convergidos que quepan en 4096 chars."""
    from .notifier import send_telegram
    poller  = get_poller(request)
    app_cfg = request.app.state.config
    tg      = app_cfg.get("telegram", {})
    token   = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")
    text    = _build_converged_text(poller, limit)
    ok      = await send_telegram(token, chat_id, "PING", "converged", {"_raw": text})
    total   = sum(1 for s in poller.snapshots.values() if s.status == "converged")
    shown   = text.count("•")
    return {"ok": ok, "shown": shown, "total": total}


@router.get("/api/status/report")
async def status_report(request: Request) -> dict:
    """Devuelve el reporte STATUS completo como texto."""
    poller  = get_poller(request)
    metrics = collect_metrics()
    text    = _build_status_report(poller, metrics)
    return {"report": text, "timestamp": datetime.now().isoformat()}


@router.post("/api/notify/status")
async def notify_status(request: Request) -> dict:
    """Envía reporte STATUS completo a Telegram ahora."""
    from .notifier import send_telegram
    poller   = get_poller(request)
    app_cfg  = request.app.state.config
    tg       = app_cfg.get("telegram", {})
    token    = tg.get("bot_token", "")
    chat_id  = tg.get("chat_id", "")
    metrics  = collect_metrics()
    text     = _build_status_report(poller, metrics)
    ok = await send_telegram(token, chat_id, "PING", "status-report", {"_raw": text})
    return {"ok": ok}


# ── Batches y control ────────────────────────────────────────────────────────

@router.get("/api/batches")
async def list_batches_endpoint(request: Request) -> dict:
    """Batches con su recuento por estado, throughput y ETA."""
    from .services.control import list_batches

    return list_batches(get_poller(request))


def _auditar(request: Request, accion: str, **campos) -> None:
    from .security import audit

    cliente = request.client.host if request.client else None
    audit(request.app.state, accion, client=cliente, **campos)


@router.post("/api/jobs/{job_id}/kill")
async def kill_job_endpoint(request: Request, job_id: str) -> dict:
    """Detiene los procesos de un job. Acción destructiva: queda auditada."""
    from .services.control import ControlError, kill_job

    job_dir = _job_dir_or_404(request, job_id)
    try:
        resultado = await kill_job(job_dir)
    except ControlError as exc:
        _auditar(request, "kill_rechazado", job_id=job_id, motivo=str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _auditar(request, "kill", job_id=job_id, pids=resultado["killed_pids"])
    return resultado


@router.post("/api/jobs/{job_id}/retry")
async def retry_job_endpoint(request: Request, job_id: str) -> dict:
    """Devuelve un job fallido a la cola para que el runner lo recoja."""
    from .services.control import ControlError, retry_job

    job_dir = _job_dir_or_404(request, job_id)
    try:
        resultado = retry_job(job_dir)
    except ControlError as exc:
        _auditar(request, "retry_rechazado", job_id=job_id, motivo=str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _auditar(request, "retry", job_id=job_id, intento=resultado["requeue_count"])
    return resultado


@router.post("/api/batches/{batch_id}/start")
async def start_batch_endpoint(request: Request, batch_id: int) -> dict:
    """Lanza el runner de un batch. Arranca un proceso: queda auditado."""
    from .services.control import ControlError, start_batch

    poller = get_poller(request)
    if not platform_caps.runner_launch_available(poller.cfg):
        detalle = (
            "Lanzar runners no está disponible aquí: falta un intérprete de "
            "Python o los scripts/ del pipeline en la raíz de datos."
        )
        _auditar(request, "start_no_disponible", batch_id=batch_id, motivo=detalle)
        raise HTTPException(status_code=501, detail=detalle)

    try:
        resultado = start_batch(poller, batch_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No existe {exc}") from exc
    except ControlError as exc:
        _auditar(request, "start_rechazado", batch_id=batch_id, motivo=str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _auditar(request, "start_batch", batch_id=batch_id, runner=resultado["runner_kind"])
    return resultado


# ── Estructuras, reportes y figuras ──────────────────────────────────────────

@router.get("/api/structures")
async def list_structures_endpoint(request: Request) -> dict:
    """Estructuras disponibles: fases de referencia, top-8 y las de cada job."""
    from .services.files import list_structures

    return {"items": list_structures(get_poller(request).runs_dir)}


@router.get("/api/structures/content")
async def structure_content(
    request: Request,
    id: str = Query(..., description="Identificador de /api/structures (repo:… o job:…)."),
) -> dict:
    """Estructura en CIF. Los structures/*.json de ASE se convierten al vuelo."""
    from .services.files import UnsafePathError, read_structure

    try:
        cif, name, metadata = read_structure(get_poller(request).runs_dir, id)
    except UnsafePathError:
        raise HTTPException(status_code=400, detail="Identificador inválido") from None
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Estructura '{id}' no encontrada") from None
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo convertir: {exc}") from exc

    return {"id": id, "name": name, "format": "cif", "content": cif, "metadata": metadata}


@router.get("/api/reports")
async def list_reports_endpoint() -> dict:
    """Reportes Markdown y galerías declaradas en los visualization_manifest.json."""
    from .services.files import list_reports

    return list_reports()


@router.get("/api/reports/document")
async def report_document(
    path: str = Query(..., description="Ruta relativa al repo, dentro de reports/ o imagenes/."),
) -> dict:
    from .services.files import UnsafePathError, read_report

    try:
        content, name = read_report(path)
    except UnsafePathError:
        raise HTTPException(status_code=400, detail="Ruta no permitida") from None
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No existe '{path}'") from None
    return {"path": path, "name": name, "content": content}


@router.get("/api/reports/figure")
async def report_figure(
    path: str = Query(..., description="Ruta relativa al repo de una figura."),
):
    """Sirve una figura. Muchas están en .gitignore y pueden no existir."""
    from starlette.responses import FileResponse

    from .services.files import UnsafePathError, resolve_figure

    try:
        file_path, mime = resolve_figure(path)
    except UnsafePathError:
        raise HTTPException(status_code=400, detail="Ruta no permitida") from None
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"'{path}' no está en disco. Regenera con scripts/generate_visualizations.py",
        ) from None
    return FileResponse(file_path, media_type=mime)


# ── Cribado HTS ──────────────────────────────────────────────────────────────

@router.get("/api/screening/config")
async def screening_config() -> dict:
    """Cotas de la cascada y disponibilidad real de cada tier."""
    from .services.screening import config_path, gates, load_generator_config, tier_availability

    try:
        cfg = load_generator_config()
    except FileNotFoundError:
        return {
            "available": False,
            "reason": f"No se encuentra {config_path()}",
            "tiers": tier_availability(),
            "gates": None,
        }
    return {"available": True, "reason": None, "tiers": tier_availability(cfg), "gates": gates(cfg)}


class ScreeningRunRequest(BaseModel):
    batch_id: int | None = None
    n_candidates: int = 200
    n_batches: int = 1
    random_seed: int | None = None
    use_mlff: bool | None = None


class ScreeningStartDftRequest(BaseModel):
    start_runner: bool = True


@router.post("/api/screening/run")
async def screening_run(request: Request, body: ScreeningRunRequest) -> dict:
    """Arranca la cascada en segundo plano y devuelve el identificador.

    Tarda minutos: se consulta el progreso en /api/screening/runs/{run_id}.
    """
    from .services.screening import start_run

    if not 1 <= body.n_candidates <= 5000:
        raise HTTPException(status_code=422, detail="candidatos por lote debe estar entre 1 y 5000")
    if not 1 <= body.n_batches <= 50:
        raise HTTPException(status_code=422, detail="lotes debe estar entre 1 y 50")
    if body.n_candidates * body.n_batches > 5000:
        raise HTTPException(status_code=422, detail="candidatos por lote × lotes no debe pasar de 5000")
    if body.random_seed is not None and not 0 <= body.random_seed <= 999_999_999:
        raise HTTPException(status_code=422, detail="semilla debe estar entre 0 y 999999999")
    if body.batch_id is not None and body.batch_id < 0:
        raise HTTPException(status_code=422, detail="batch_id debe ser no negativo")

    try:
        run = start_run(
            batch_id=body.batch_id,
            n_candidates=body.n_candidates,
            n_batches=body.n_batches,
            random_seed=body.random_seed,
            use_mlff=body.use_mlff,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=501, detail=f"Falta la configuración: {exc}") from exc

    _auditar(
        request,
        "screening_run",
        batch_id=run.batch_id,
        random_seed=run.random_seed,
        lotes=run.n_batches,
        n=run.n_requested,
    )
    resultado = run.as_dict()
    resultado.pop("items", None)
    resultado.pop("dropped", None)
    return resultado


@router.get("/api/screening/runs")
async def screening_runs() -> dict:
    from .services.screening import list_runs

    return {"items": list_runs()}


@router.get("/api/screening/runs/{run_id}")
async def screening_run_detail(
    run_id: str,
    limit: int = Query(200, ge=1, le=500, description="Filas del ranking a devolver."),
) -> dict:
    from .services.screening import get_run

    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Ejecución '{run_id}' no encontrada")

    resultado = run.as_dict()
    resultado["items"] = resultado["items"][:limit]
    resultado["n_items_total"] = len(run.items)
    return resultado


@router.post("/api/screening/runs/{run_id}/start-dft")
async def screening_start_dft(
    request: Request,
    run_id: str,
    body: ScreeningStartDftRequest,
) -> dict:
    """Prepara los seleccionados del cribado como jobs DFT y puede lanzar el runner."""
    from .services.control import ControlError
    from .services.screening import start_dft_for_run

    try:
        result = start_dft_for_run(
            get_poller(request),
            run_id,
            start_runner=body.start_runner,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Ejecución '{run_id}' no encontrada") from exc
    except (RuntimeError, ControlError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ImportError as exc:
        # Preparar los jobs construye las estructuras en proceso y necesita
        # `ase`. Si falta —porque el empaquetado la dejó fuera— el usuario veía
        # un 500 sin explicación. Mejor decir exactamente qué falta.
        falta = getattr(exc, "name", None) or str(exc)
        raise HTTPException(
            status_code=501,
            detail=(f"Esta instalación no puede preparar jobs DFT: falta el módulo "
                    f"'{falta}'. Prepara el lote desde el repositorio."),
        ) from exc

    _auditar(
        request,
        "screening_start_dft",
        run_id=run_id,
        batch_id=result["batch_id"],
        n_prepared=result["n_prepared"],
        runner_launched=result["runner_launched"],
    )
    return result


# ── Descubrimiento autónomo ─────────────────────────────────────────────────

class DiscoveryInitRequest(BaseModel):
    reset: bool = False


class DiscoverySpaceRequest(BaseModel):
    A_sites: list[str] | None = None
    B_sites: list[str] | None = None
    X_sites: list[str] | None = None
    modes: dict[str, bool] | None = None
    min_fraction: float | None = None
    max_fraction: float | None = None
    fraction_step: float | None = None
    include_multi_mixed: bool | None = None
    dft_per_round: int | None = None


def _discovery_space_payload(body: DiscoverySpaceRequest) -> dict:
    if hasattr(body, "model_dump"):
        return body.model_dump(exclude_none=True)
    return body.dict(exclude_none=True)


class DiscoveryRunRequest(BaseModel):
    start_runner: bool = True
    dry_run: bool = False
    use_mlff: bool | None = None
    max_rounds: int | None = None


@router.get("/api/discovery/config")
async def discovery_config() -> dict:
    """Configuración efectiva del espacio químico del protocolo."""
    from .services.discovery import current_config

    return current_config()


@router.post("/api/discovery/config/preview")
async def discovery_config_preview(body: DiscoverySpaceRequest | None = None) -> dict:
    """Cuenta el espacio químico que produciría una configuración."""
    from .services.discovery import preview_config

    body = body or DiscoverySpaceRequest()
    try:
        return preview_config(_discovery_space_payload(body))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/discovery/config")
async def discovery_config_save(request: Request, body: DiscoverySpaceRequest) -> dict:
    """Guarda el override editable del espacio químico."""
    from .services.discovery import save_config

    try:
        result = save_config(_discovery_space_payload(body))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _auditar(request, "discovery_config_save", override=result.get("override_path"))
    return result


@router.get("/api/discovery/status")
async def discovery_status() -> dict:
    """Estado persistente del loop autónomo de descubrimiento."""
    from .services.discovery import status

    return status()


@router.post("/api/discovery/init")
async def discovery_init(request: Request, body: DiscoveryInitRequest | None = None) -> dict:
    """Enumera el espacio químico finito y crea el ledger persistente."""
    from .services.discovery import init

    body = body or DiscoveryInitRequest()
    try:
        result = init(reset=body.reset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=501, detail=f"Falta la configuración: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _auditar(request, "discovery_init", reset=body.reset)
    return result


@router.post("/api/discovery/run")
async def discovery_run(request: Request, body: DiscoveryRunRequest | None = None) -> dict:
    """Arranca el loop autónomo en segundo plano."""
    from .services.discovery import start

    body = body or DiscoveryRunRequest()
    if body.max_rounds is not None and body.max_rounds < 1:
        raise HTTPException(status_code=422, detail="max_rounds debe ser positivo")
    try:
        result = start(
            start_runner=body.start_runner,
            dry_run=body.dry_run,
            use_mlff=body.use_mlff,
            max_rounds=body.max_rounds,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=501, detail=f"Falta un recurso: {exc}") from exc
    _auditar(
        request,
        "discovery_run",
        start_runner=body.start_runner,
        dry_run=body.dry_run,
        use_mlff=body.use_mlff,
        max_rounds=body.max_rounds,
    )
    return result


@router.post("/api/discovery/pause")
async def discovery_pause(request: Request) -> dict:
    from .services.discovery import pause

    result = pause()
    _auditar(request, "discovery_pause")
    return result


@router.post("/api/discovery/stop")
async def discovery_stop(request: Request) -> dict:
    """Termina el subproceso del bucle.

    `pause` pide al bucle que pare en el siguiente punto de control, que puede
    tardar una ronda entera; esto lo corta. Los cálculos DFT ya lanzados siguen
    su curso: son procesos aparte, igual que antes.
    """
    from .services.discovery import stop

    result = stop()
    _auditar(request, "discovery_stop")
    return result


@router.post("/api/discovery/resume")
async def discovery_resume(request: Request) -> dict:
    from .services.discovery import resume

    result = resume()
    _auditar(request, "discovery_resume")
    return result


@router.get("/api/discovery/frontier")
async def discovery_frontier(
    limit: int = Query(100, ge=1, le=500, description="Candidatos de la frontera a devolver."),
) -> dict:
    from .services.discovery import frontier

    return frontier(limit=limit)


@router.post("/api/discovery/export")
async def discovery_export(request: Request) -> dict:
    from .services.discovery import export

    result = export()
    _auditar(request, "discovery_export", report=result.get("report"))
    return result


# ── Arranque de un clic ──────────────────────────────────────────────────────

class QuickStartRequest(BaseModel):
    """Todo opcional: el sentido del boton es no tener que decidir nada."""

    max_rounds: int | None = None
    use_mlff: bool | None = None
    dry_run: bool = False
    con_benchmark: bool = True


@router.get("/api/quickstart/status")
async def quickstart_status() -> dict:
    """Lo que dejo el ultimo arranque y como va el protocolo ahora."""
    from .services.quickstart import estado

    return estado()


@router.get("/api/quickstart/probe")
async def quickstart_probe() -> dict:
    """Sondeo de la maquina sin arrancar nada: para ver el reparto antes.

    Separado de `POST /api/quickstart` a proposito: mirar cuantos trabajos
    saldrian no deberia lanzar dias de calculo.
    """
    from .services.hwprobe import sondear

    return sondear()


@router.post("/api/quickstart")
async def quickstart_run(request: Request, body: QuickStartRequest | None = None) -> dict:
    """Mide la maquina, fija el reparto, comprueba requisitos y arranca."""
    from .services.quickstart import arrancar

    body = body or QuickStartRequest()
    if body.max_rounds is not None and body.max_rounds < 1:
        raise HTTPException(status_code=422, detail="max_rounds debe ser positivo")

    resultado = arrancar(
        max_rounds=body.max_rounds,
        use_mlff=body.use_mlff,
        dry_run=body.dry_run,
        con_benchmark=body.con_benchmark,
    )
    _auditar(
        request,
        "quickstart",
        arrancado=resultado.get("arrancado"),
        motivo=resultado.get("motivo"),
        reparto=resultado.get("reparto"),
    )
    return resultado


class SetupInstallRequest(BaseModel):
    target: str
    cuda: bool = False
    recreate: bool = False
    env_name: str | None = None
    distro: str | None = None


def _setup_opciones(body: SetupInstallRequest) -> dict:
    """Solo `mlff` acepta opciones; los grupos pip no tienen ninguna."""
    if body.target != "mlff":
        return {}
    return {
        "cuda": body.cuda,
        "recrear": body.recreate,
        "env_name": body.env_name,
        "distro": body.distro,
    }


@router.get("/api/setup/status")
async def setup_status(
    fast: bool = Query(False, description="Omite la sonda MLFF, que lanza un proceso."),
) -> dict:
    """Qué entornos funcionan en esta máquina y qué falta para los que no."""
    from .services.setup import status

    return status(fast=fast)


@router.post("/api/setup/plan")
async def setup_plan(body: SetupInstallRequest) -> dict:
    """Los comandos que se ejecutarían, sin ejecutar ninguno."""
    from .services.setup import plan

    try:
        return plan(body.target, **_setup_opciones(body))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/setup/install")
async def setup_install(request: Request, body: SetupInstallRequest) -> dict:
    """Instala en segundo plano lo que falta para `target`."""
    from .services.setup import start_install

    try:
        result = start_install(body.target, **_setup_opciones(body))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Ya hay una instalación, o el protocolo está cribando: las dos son
        # colisiones de estado, no errores del cliente.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _auditar(request, "setup_install", target=body.target, cuda=body.cuda,
             recreate=body.recreate)
    return result


@router.get("/api/setup/job")
async def setup_job() -> dict:
    """Estado y log de la instalación en curso (o de la última)."""
    from .services.setup import job

    return job()


@router.get("/api/activity")
async def activity(request: Request) -> dict:
    """Qué está haciendo el sistema ahora y cuánto le queda."""
    from .services.activity import describe

    return describe(get_poller(request))


# ── Calibración de rendimiento ───────────────────────────────────────────────

class BenchRunRequest(BaseModel):
    mode: Literal["quick", "full"] = "quick"
    force: bool = False


@router.get("/api/bench")
async def bench_status(request: Request) -> dict:
    """Máquina detectada, última calibración y barrido en curso si lo hay."""
    from .services.bench import status

    return status(get_poller(request))


@router.post("/api/bench/run")
async def bench_run(request: Request, body: BenchRunRequest) -> dict:
    """Lanza el barrido slots×cores. Arranca procesos: queda auditado."""
    from .services.bench import start
    from .services.control import ControlError

    try:
        resultado = start(get_poller(request), mode=body.mode, force=body.force)
    except ControlError as exc:
        _auditar(request, "bench_rechazado", modo=body.mode, motivo=str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _auditar(request, "bench_run", modo=body.mode, pid=resultado["pid"])
    return resultado


@router.post("/api/bench/cancel")
async def bench_cancel(request: Request) -> dict:
    """Detiene el barrido en marcha. Acción destructiva: queda auditada."""
    from .services.bench import cancel
    from .services.control import ControlError

    try:
        resultado = cancel()
    except ControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _auditar(request, "bench_cancel", pids=resultado["killed_pids"])
    return resultado


# ── Candidatos y ML ──────────────────────────────────────────────────────────

@router.get("/api/candidates")
async def list_candidates(
    request: Request,
    q: str | None = Query(None, description="Subcadena en la fórmula."),
    generation_mode: str | None = Query(None, description="pure/A_mixed/B_mixed/X_mixed, coma."),
    b_family: str | None = Query(None, description="Familia del sitio B, coma."),
    halide: str | None = Query(None, description="Haluro dominante, coma."),
    sort: str = Query("score", pattern="^(score|formula|tolerance_t|oct_factor)$"),
    desc: bool = True,
    verified_only: bool = Query(True, description="Solo viables, con DFT terminado y buen score PV."),
    pv_min: float = Query(0.5, ge=0.0, le=1.0, description="Score fotovoltaico mínimo."),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Candidatos del generador BUHO.

    `source` indica de dónde salieron: el CSV del generador si está disponible,
    o los metadata.json de los jobs como alternativa.
    """
    from .services.candidates import query_candidates

    return query_candidates(
        get_poller(request).runs_dir,
        solo_verificados=verified_only,
        umbral_pv=pv_min,
        q=q,
        generation_mode=generation_mode,
        b_family=b_family,
        halide=halide,
        sort=sort,
        desc=desc,
        limit=limit,
        offset=offset,
    )


@router.get("/api/models")
async def list_models() -> dict:
    """Métricas de los surrogates y estado de carga del modelo de bandgap."""
    from .services.ml import model_metrics

    return model_metrics()


class PredictRequest(BaseModel):
    A: str
    B: str
    X: str
    a_lat: float | None = None
    e_mace_ev_atom: float | None = None
    band_gap_gga_ev: float | None = None
    eform_ev_atom: float | None = None
    material: str | None = None


@router.post("/api/ml/predict")
async def ml_predict(body: PredictRequest) -> dict:
    """Bandgap predicho con incertidumbre bootstrap para una composición ABX3."""
    from .services.ml import SurrogateUnavailableError, predict

    try:
        return predict(
            body.A,
            body.B,
            body.X,
            a_lat=body.a_lat,
            e_mace_ev_atom=body.e_mace_ev_atom,
            band_gap_gga_ev=body.band_gap_gga_ev,
            eform_ev_atom=body.eform_ev_atom,
            material=body.material,
        )
    except SurrogateUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        # Composición fuera del espacio químico del modelo.
        raise HTTPException(status_code=422, detail=f"Composición no soportada: {exc}") from exc


@router.get("/api/ml/top8")
async def ml_top8() -> dict:
    """Predicción ML frente a DFT y experimento para los 8 candidatos top."""
    from .services.ml import top8_reference

    return {"items": top8_reference()}


@router.get("/api/ml/parity")
async def ml_parity(request: Request,
                    limit: int = Query(8, ge=1, le=50)) -> dict:
    """Los mejores del lote según el predictor, frente al DFT ya calculado.

    Sustituye a la comparación contra referencias de literatura: aquí ambas
    columnas son gap PBE, así que la diferencia mide solo el predictor.
    """
    from .services.activity import runners_activos
    from .services.control import _raiz_batches
    from .services.ml import parity_from_batch

    poller = get_poller(request)

    # Se prefiere el lote que se está calculando; si no hay ninguno, el más
    # reciente con jobs terminados.
    batch = next((r["batch"] for r in runners_activos() if r.get("batch")), None)
    if batch is None:
        raiz = _raiz_batches(poller)
        lotes = sorted((d for d in raiz.glob("batch_*") if d.is_dir()),
                       key=lambda d: d.stat().st_mtime, reverse=True)
        batch = next((d for d in lotes if any(d.glob("*/status.json"))), None)

    if batch is None:
        return {"batch": None, "items": [], "n_converged": 0, "n_with_gap": 0,
                "error": "No hay ningún lote con resultados."}

    return parity_from_batch(batch, limit=limit)


@router.post("/api/notify/test")
async def notify_test(request: Request) -> dict:
    from .notifier import send_test
    app_cfg  = request.app.state.config
    tg       = app_cfg.get("telegram", {})
    token    = tg.get("bot_token", "")
    chat_id  = tg.get("chat_id", "")
    ok       = await send_test(token, chat_id)
    return {"ok": ok, "message": "Mensaje enviado" if ok else "Fallo — revisa bot_token y chat_id"}
