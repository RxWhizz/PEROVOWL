"""Calibración de rendimiento desde el monitor.

Un barrido dura horas y lanza procesos GPAW, así que corre como subproceso
desacoplado —igual que el runner— y no como hilo del servidor: cerrar la app no
debe tirar una medición de tres horas. El avance se lee de un archivo que el
propio barrido escribe.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from .. import paths, platform_caps
from ..services.control import ControlError

MODOS = {"quick", "full"}


def _progress_file() -> Path:
    return paths.resolve_data("data/bench/progress.json")


def _script() -> Path:
    """El barrido, en la raiz de datos.

    En una instalacion binaria llega ahi porque `paths.materializar_pipeline`
    lo copia del bundle al arrancar; desde el repositorio ya esta. Antes solo
    existia en el repositorio, asi que `can_run` era False en todo binario y la
    calibracion no se podia lanzar.
    """
    return paths.data_root() / "scripts" / "bench_machine.py"


def _leer_progreso() -> dict[str, Any]:
    ruta = _progress_file()
    if not ruta.is_file():
        return {}
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return datos if isinstance(datos, dict) else {}


def _vivo(pid: int | None) -> bool:
    """Si el barrido sigue corriendo de verdad.

    El archivo de progreso puede quedar en «running» si el proceso murió de
    golpe; sin comprobar el PID la interfaz mostraría un barrido fantasma para
    siempre y nunca dejaría lanzar otro.
    """
    if not pid:
        return False
    try:
        proc = psutil.Process(int(pid))
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        return "bench_machine" in " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def machine_info() -> dict[str, Any]:
    """Descripción de la máquina, o el motivo de no poder obtenerla."""
    try:
        from buho.bench.machine import budgets_for, detect, ram_limit_gb, splits_for
    except ImportError as exc:
        return {"available": False, "reason": f"falta el módulo de benchmark: {exc}"}

    m = detect()
    presupuestos = budgets_for(m)
    return {
        "available": True,
        **m.as_dict(),
        "description": m.describe(),
        "ram_limit_gb": ram_limit_gb(m),
        "budgets": presupuestos,
        "n_splits_quick": sum(len(splits_for(b, max_splits=3)) for b in presupuestos),
        "n_splits_full": sum(len(splits_for(b)) for b in presupuestos),
    }


def calibration_info() -> dict[str, Any] | None:
    """Lo que se midió la última vez en esta máquina."""
    try:
        from buho.bench import calibration as calib
        from buho.bench.machine import detect
    except ImportError:
        return None

    cal = calib.load(detect(), paths.data_root())
    if cal is None:
        return None
    return {
        "best_slots": cal.best.slots,
        "best_cores": cal.best.cores,
        "budget": cal.budget,
        "throughput": cal.throughput,
        "peak_ram_gb": cal.peak_ram_gb,
        "measured_at": cal.measured_at,
        "n_results": len(cal.results),
    }


def busy_reasons() -> list[str]:
    """Cálculos en marcha que falsearían la medición."""
    culpables = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or ())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not cmd or "bench_machine" in cmd or "_bench" in cmd:
            continue
        if any(p in cmd for p in ("buho_relax_runner", "python input.py", "gpaw-python")):
            culpables.append(cmd[:120])
    return culpables


def status(poller=None) -> dict[str, Any]:
    """Estado completo para la interfaz."""
    prog = _leer_progreso()
    corriendo = prog.get("status") == "running" and _vivo(prog.get("pid"))

    estado = prog.get("status", "idle")
    if prog.get("status") == "running" and not corriendo:
        estado = "interrupted"   # murió sin cerrar el archivo

    cfg = getattr(poller, "cfg", {}) or {}
    return {
        "status": estado,
        "running": corriendo,
        "done": int(prog.get("done", 0)),
        "total": int(prog.get("total", 0)),
        "current": prog.get("current"),
        "error": prog.get("error"),
        "updated_at": prog.get("updated_at"),
        "results": prog.get("results", []),
        "machine": machine_info(),
        "calibration": calibration_info(),
        "busy": busy_reasons(),
        "can_run": platform_caps.runner_launch_available(cfg) and _script().is_file(),
        "configured_slots": int(cfg.get("runner_slots", 0) or 0),
        "configured_cores": int(cfg.get("runner_cores", 0) or 0),
    }


def start(poller, *, mode: str = "quick", force: bool = False) -> dict[str, Any]:
    """Lanza el barrido en segundo plano."""
    if mode not in MODOS:
        raise ControlError(f"Modo desconocido '{mode}'. Usa 'quick' o 'full'.")

    prog = _leer_progreso()
    if prog.get("status") == "running" and _vivo(prog.get("pid")):
        raise ControlError("Ya hay un barrido en marcha.")

    script = _script()
    if not script.is_file():
        raise ControlError(f"No se encuentra {script}. ¿Instalación incompleta?")

    interprete = platform_caps.runner_python(getattr(poller, "cfg", {}) or {})
    if interprete is None:
        raise ControlError(
            "No hay un intérprete de Python con el que lanzar el barrido. "
            "La app de escritorio necesita el repositorio para medir."
        )

    ocupada = busy_reasons()
    if ocupada and not force:
        raise ControlError(
            f"Hay {len(ocupada)} cálculo(s) en marcha. Medir ahora daría un "
            "t/iter inflado y recomendaría menos slots de los que aguanta la "
            "máquina. Espera a que terminen o fuerza la medición."
        )

    cmd = [interprete, str(script)]
    if mode == "quick":
        cmd.append("--quick")
    if force:
        cmd.append("--force")

    raiz = paths.data_root()
    log = raiz / "data" / "bench" / "bench.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    salida = open(log, "a", encoding="utf-8")
    salida.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} modo={mode} =====\n")
    salida.flush()

    try:
        proc = subprocess.Popen(cmd, cwd=str(raiz), stdout=salida, stderr=salida,
                                start_new_session=True)
    except OSError as exc:
        salida.close()
        raise ControlError(f"No se pudo lanzar el barrido: {exc}") from exc

    # Marca inmediata: el script tarda unos segundos en escribir su progreso y
    # sin esto la interfaz diría «inactivo» justo después de pulsar.
    _progress_file().parent.mkdir(parents=True, exist_ok=True)
    _progress_file().write_text(json.dumps({
        "status": "running", "pid": proc.pid, "done": 0, "total": 0,
        "current": None, "results": [], "error": None, "updated_at": time.time(),
    }, indent=2) + "\n", encoding="utf-8")

    return {"started": True, "pid": proc.pid, "mode": mode, "log": str(log)}


def cancel() -> dict[str, Any]:
    """Detiene el barrido en marcha y sus hijos."""
    prog = _leer_progreso()
    pid = prog.get("pid")
    if not (prog.get("status") == "running" and _vivo(pid)):
        raise ControlError("No hay ningún barrido en marcha.")

    muertos = []
    try:
        padre = psutil.Process(int(pid))
        for hijo in padre.children(recursive=True):
            try:
                hijo.kill()
                muertos.append(hijo.pid)
            except psutil.Error:
                pass
        padre.kill()
        muertos.append(padre.pid)
    except psutil.Error as exc:
        raise ControlError(f"No se pudo detener el barrido: {exc}") from exc

    prog["status"] = "cancelled"
    prog["updated_at"] = time.time()
    try:
        _progress_file().write_text(json.dumps(prog, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return {"cancelled": True, "killed_pids": muertos}
