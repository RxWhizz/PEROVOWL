"""Background poller: escanea status.json + logs GPAW/FIRE, emite eventos."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from buho.phase2_force.self_heal import (
    evaluate_u_oscillation_self_heal,
    status_update_for_u_decision,
    write_u_scan_decision,
)

from . import paths, platform_caps
from .models import JobStatus, JobStatusLiteral, StatsResponse, WsEvent
from .notifier import send_telegram

log = logging.getLogger(__name__)

# Regexes reutilizadas de buho_monitor.py / gpaw_monitor.py
_OPTIMIZER_RE = re.compile(
    r"(?:FIRE|BFGS|LBFGS):\s+(\d+)\s+(\d{2}:\d{2}:\d{2})\s+([-\d.]+)\s+([\d.]+)"
)
_SCF_TS_RE = re.compile(
    r"\|iter:\s+(\d+)\|\s+(\d{2}):(\d{2}):(\d{2})\s*\|\s*([-\d.]+)"
)
_SCF_RE = re.compile(r"iter:\s+(\d+)\s+(\d{2}:\d{2}:\d{2})")
_ENERGY_RE = re.compile(r"Total:\s+([-\d.]+)")
_CONV_RE = re.compile(r"Converged after|scf cycle converged", re.I)
_ERROR_RE = re.compile(r"Traceback|MemoryError|Segfault|Killed", re.I)


# ── Helpers reutilizados ──────────────────────────────────────────────────────

def _telegram_bot_token(cfg: dict) -> str:
    """Token del bot: primero el entorno, y el YAML solo como respaldo.

    Las credenciales de Telegram vivían únicamente en `monitor.yaml`. Ese
    fichero se edita a mano y se copia al directorio XDG al instalar, así que
    es el candidato natural a colarse en un commit o en un artefacto. Con esto
    el sitio recomendado pasa a ser el `.env`, que está gitignorado, y el YAML
    puede quedarse con los huecos vacíos.
    """
    return os.environ.get("DFT_MONITOR_TELEGRAM_BOT_TOKEN", "") or str(
        cfg.get("bot_token", "") or ""
    )


def _telegram_chat_id(cfg: dict) -> str:
    """Chat de destino: mismo orden que el token."""
    return os.environ.get("DFT_MONITOR_TELEGRAM_CHAT_ID", "") or str(
        cfg.get("chat_id", "") or ""
    )


def _is_pid_alive(pid: int) -> bool:
    return psutil.pid_exists(pid)


def _proc_rss_mb(pid: int) -> int | None:
    try:
        return psutil.Process(pid).memory_info().rss // (1024 * 1024)
    except (psutil.Error, ValueError):
        return None


def _fmt_elapsed(iso_start: str) -> float | None:
    try:
        start = datetime.fromisoformat(iso_start.replace("Z", "+00:00")).replace(tzinfo=None)
        return round((datetime.now() - start).total_seconds() / 60, 1)
    except Exception:
        return None


def _read_status(job_dir: Path) -> dict:
    import json
    p = job_dir / "status.json"
    try:
        data = json.loads(p.read_text())
    except Exception:
        data = {"status": "unknown", "candidate_id": job_dir.name}
    # Si no hay fórmula en status.json, buscarla en metadata.json
    if not data.get("formula"):
        meta = job_dir / "metadata.json"
        try:
            data["formula"] = json.loads(meta.read_text()).get("formula", job_dir.name)
        except Exception:
            data["formula"] = job_dir.name
    return data


def _write_status_update(job_dir: Path, update: dict[str, Any]) -> None:
    p = job_dir / "status.json"
    data = _read_status(job_dir)
    data.update(update)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def _parse_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _phase2_label_log_paths(job_dir: Path, st: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for label in st.get("labels_expected") or []:
        rel = label.get("relative_dir")
        if rel:
            paths.append(job_dir / rel / "r2scan.txt")
    paths.extend(job_dir.glob("r2scan/r2scan.txt"))
    paths.extend(job_dir.glob("u_scan/*/r2scan.txt"))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _phase2_scf_progress(job_dir: Path, st: dict[str, Any]) -> dict[str, Any]:
    started_epoch = _parse_epoch(st.get("started_at") or st.get("start_time"))
    newest: dict[str, Any] | None = None
    total_iters = 0
    now = time.time()
    for path in _phase2_label_log_paths(job_dir, st):
        if not path.exists():
            continue
        stat = path.stat()
        if started_epoch is not None and stat.st_mtime < started_epoch - 120:
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            text = ""
        matches = _SCF_RE.findall(text)
        last_iter = int(matches[-1][0]) if matches else 0
        total_iters += last_iter
        item = {
            "relative_path": str(path.relative_to(job_dir)),
            "mtime": stat.st_mtime,
            "age_min": round((now - stat.st_mtime) / 60, 1),
            "last_iter": last_iter,
            "size": stat.st_size,
        }
        if newest is None or stat.st_mtime > float(newest["mtime"]):
            newest = item
    return {"newest": newest, "total_iters": total_iters}


# Segundos entre el intento educado y el forzado.
_ESPERA_TERMINACION = 3


def _salida_desligada(batch_dir: Path):
    """Descriptores propios para el runner, en vez de heredar los del monitor.

    `start_new_session` desliga el *grupo de procesos*, pero no los descriptores:
    el runner seguía heredando el stdout/stderr del motor, que en la app de
    escritorio son tuberías que lee Flutter. Al cerrar la app la tubería se
    rompía y el runner moría de SIGPIPE en su siguiente `print` —el log de
    batch_765153 termina en un LAUNCH sin su DONE, con 42 jobs sin hacer—.
    Escribiendo a un archivo, el runner sobrevive a la app que lo lanzó.
    """
    try:
        batch_dir.mkdir(parents=True, exist_ok=True)
        return open(batch_dir / "runner_console.log", "a", encoding="utf-8")
    except OSError:
        return subprocess.DEVNULL


def _grupo_propio() -> dict[str, Any]:
    """Argumentos de Popen para desligar el hijo del monitor.

    En POSIX es `start_new_session`; en Windows, un flag de creación. Sin esto,
    parar el monitor se llevaría por delante a los runners.
    """
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _job_processes(job_dir: Path, root_pid: int | None = None) -> set[int]:
    """PIDs que trabajan en este job: los que tienen ahí su cwd, más el árbol
    de descendientes del proceso raíz.

    Vía psutil en vez de leyendo /proc a mano: funciona igual en Windows y deja
    de depender del formato de `PPid:`.
    """
    raiz = str(job_dir.resolve())
    pids: set[int] = set()

    for proc in psutil.process_iter(["pid"]):
        try:
            cwd = proc.cwd()
        except (psutil.Error, OSError):
            continue  # el proceso murió, o no es nuestro
        if cwd and (cwd == raiz or cwd.startswith(raiz + os.sep)):
            pids.add(proc.pid)

    if isinstance(root_pid, int):
        pids.add(root_pid)
        try:
            pids.update(h.pid for h in psutil.Process(root_pid).children(recursive=True))
        except psutil.Error:
            pass

    return pids


def _kill_job_processes(job_dir: Path, root_pid: int | None = None) -> list[int]:
    """SIGTERM y, a los que sobrevivan, SIGKILL. Devuelve los PID afectados.

    `Process.terminate()`/`kill()` son multiplataforma; sustituyen a killpg
    sobre el grupo de procesos. Se aplican a todo el árbol de descendientes, que
    es lo que antes conseguía el grupo con los procesos MPI.
    """
    pids = _job_processes(job_dir, root_pid)

    procesos: list[psutil.Process] = []
    for pid in sorted(pids):
        try:
            procesos.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            pass

    for proc in procesos:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    _, vivos = psutil.wait_procs(procesos, timeout=_ESPERA_TERMINACION)
    for proc in vivos:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return sorted(pids)


# ── Log parsers ───────────────────────────────────────────────────────────────

def _parse_relax_log(job_dir: Path) -> dict:
    """Extrae historial FIRE/BFGS del relax.log."""
    log_file = job_dir / "relax.log"
    if not log_file.exists():
        return {}
    try:
        lines = log_file.read_text(errors="replace").splitlines()
        steps, energies, fmaxes = [], [], []
        for line in lines:
            m = _OPTIMIZER_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                energies.append(float(m.group(3)))
                fmaxes.append(float(m.group(4)))
        if not steps:
            return {}
        return {
            "step": steps[-1],
            "energy": energies[-1],
            "fmax": fmaxes[-1],
            "energy_history": energies,
            "fmax_history": fmaxes,
            "step_type": "FIRE",
        }
    except Exception:
        return {}


def _parse_scf_log(job_dir: Path) -> dict:
    """Extrae iteraciones SCF y energías del r2scan.txt / gpaw.txt."""
    for name in ("r2scan.txt", "gpaw.txt"):
        txt = job_dir / name
        if not txt.exists():
            continue
        try:
            content = txt.read_text(errors="replace")
            matches = _SCF_TS_RE.findall(content)
            if matches:
                iters   = [int(m[0]) for m in matches]
                energies = [float(m[4]) for m in matches]

                times = []
                for m in matches:
                    h, mn, s = int(m[1]), int(m[2]), int(m[3])
                    times.append(h * 3600 + mn * 60 + s)
                deltas = [(times[i] - times[i-1]) % 86400 for i in range(1, len(times))]
                t_iter = round(sum(deltas) / len(deltas), 1) if deltas else None
                eta_iters = max(0, 30 - iters[-1])
                eta_min = round(eta_iters * t_iter / 60, 1) if t_iter else None

                return {
                    "scf_iter": iters[-1],
                    "energy": energies[-1],
                    "energy_history": energies,
                    "scf_iter_history": iters,
                    "t_iter_s": t_iter,
                    "eta_min": eta_min,
                    "step_type": "SCF",
                }
            # Fallback sin timestamp
            scf_iters = _SCF_RE.findall(content)
            n_scf = int(scf_iters[-1][0]) if scf_iters else 0
            e_match = _ENERGY_RE.search(content)
            return {
                "scf_iter": n_scf,
                "energy": float(e_match.group(1)) if e_match else None,
                "step_type": "SCF",
            }
        except Exception:
            continue
    return {}


def _detect_oscillation(energies: list[float], window: int, threshold: float) -> bool:
    if len(energies) < window:
        return False
    recent = energies[-window:]
    try:
        return statistics.stdev(recent) > threshold
    except statistics.StatisticsError:
        return False


def _log_mtime_minutes_ago(job_dir: Path) -> float | None:
    """Minutos desde que el último log fue modificado."""
    for name in ("relax.log", "r2scan.txt", "gpaw.txt"):
        p = job_dir / name
        if p.exists():
            age = time.time() - p.stat().st_mtime
            return round(age / 60, 1)
    return None


# ── Job snapshot ─────────────────────────────────────────────────────────────

def snapshot_job(job_dir: Path, cfg: dict) -> StatsResponse:
    """Construye un StatsResponse completo para un job_dir."""
    st = _read_status(job_dir)
    raw_status: str = st.get("status", "unknown")
    pid: int | None = st.get("pid")
    formula: str = st.get("formula", job_dir.name)
    start_time: str | None = st.get("start_time")
    elapsed = _fmt_elapsed(start_time) if start_time else st.get("elapsed_min")

    relax  = _parse_relax_log(job_dir)
    scf    = _parse_scf_log(job_dir)

    energy_history = relax.get("energy_history") or scf.get("energy_history") or []
    fmax_history   = relax.get("fmax_history", [])
    scf_iters      = scf.get("scf_iter_history", [])

    is_oscillating = _detect_oscillation(
        energy_history,
        window=cfg.get("oscillation_window", 10),
        threshold=cfg.get("oscillation_energy_std_ev", 0.05),
    )

    # Determina status efectivo
    effective_status: JobStatusLiteral = raw_status  # type: ignore[assignment]
    stall_minutes: float | None = None

    if raw_status == "running" and pid:
        if not _is_pid_alive(pid):
            effective_status = "stopped"  # type: ignore[assignment]
        else:
            stall_min = _log_mtime_minutes_ago(job_dir)
            stall_threshold = cfg.get("stall_minutes", 10)
            if stall_min is not None and stall_min > stall_threshold:
                effective_status = "stalled"  # type: ignore[assignment]
                stall_minutes = stall_min
            elif is_oscillating:
                effective_status = "oscillating"  # type: ignore[assignment]

    return StatsResponse(
        job_id=job_dir.name,
        formula=formula,
        status=effective_status,
        pid=pid,
        start_time=start_time,
        elapsed_min=elapsed,
        mpi_cores=st.get("mpi_cores"),
        energy_history=energy_history,
        fmax_history=fmax_history,
        scf_iter_history=scf_iters,
        n_fire_steps=relax.get("step", 0),
        n_scf_iters=scf.get("scf_iter", 0),
        is_oscillating=is_oscillating,
        stall_minutes=stall_minutes,
        final_energy_ev=st.get("final_energy_eV"),
    )


def ping_job(job_dir: Path) -> dict:
    """Lectura instantánea (para el endpoint /ping)."""
    st = _read_status(job_dir)
    pid: int | None = st.get("pid")
    alive = _is_pid_alive(pid) if pid else False

    relax = _parse_relax_log(job_dir)
    scf   = _parse_scf_log(job_dir)

    current_step = relax.get("step") or scf.get("scf_iter")
    step_type    = relax.get("step_type") or scf.get("step_type") or "?"
    energy       = relax.get("energy") or scf.get("energy")
    fmax         = relax.get("fmax")
    rss          = _proc_rss_mb(pid) if pid and alive else None

    # Última línea del log más reciente
    last_line = None
    for name in ("relax.log", "r2scan.txt", "gpaw.txt"):
        p = job_dir / name
        if p.exists():
            try:
                lines = p.read_text(errors="replace").splitlines()
                last_line = next((l for l in reversed(lines) if l.strip()), None)
                break
            except Exception:
                pass

    return {
        "alive": alive,
        "current_step": current_step,
        "step_type": step_type,
        "energy_ev": energy,
        "fmax_ev_ang": fmax,
        "memory_rss_mb": rss,
        "t_iter_s": scf.get("t_iter_s"),
        "eta_min": scf.get("eta_min"),
        "log_last_line": last_line,
        "status": st.get("status", "unknown"),
    }


# ── Background poller ─────────────────────────────────────────────────────────

# Reintento del orquestador de active learning tras un fallo: espera
# exponencial desde 5 min hasta 1 h, y se rinde tras 5 intentos seguidos.
ORCHESTRATOR_BACKOFF_S = 300.0
ORCHESTRATOR_BACKOFF_MAX_S = 3600.0
ORCHESTRATOR_MAX_FALLOS = 5


def orchestrator_log() -> Path:
    """Archivo donde se acumula el stderr del orquestador."""
    return paths.config_dir() / "orchestrator.log"


_BATCH_RE = re.compile(r"^batch_\d+$")


class DFTPoller:
    def __init__(self, runs_dir: Path, cfg: dict, telegram_cfg: dict):
        self.runs_dir    = runs_dir
        self.cfg         = cfg
        self.telegram    = telegram_cfg
        self._prev_states: dict[str, str] = {}
        self._event_queue: asyncio.Queue[WsEvent] = asyncio.Queue(maxsize=256)
        self._snapshots: dict[str, StatsResponse] = {}
        self.last_poll_at: float | None = None

    @property
    def event_queue(self) -> asyncio.Queue[WsEvent]:
        return self._event_queue

    @property
    def snapshots(self) -> dict[str, StatsResponse]:
        return self._snapshots

    def _runner_kind(self) -> str:
        return str(self.cfg.get("runner_kind", "relax"))

    def _auto_advance(self) -> bool:
        """Si el monitor puede avanzar el pipeline por su cuenta.

        Al detectar un lote terminado, el poller lanza el runner del siguiente
        o dispara el orquestador de active learning (reentrenar + generar lote).
        Eso es lo que sostiene la operación desatendida, pero significa que
        *abrir* el monitor apuntando a un lote acabado muta el pipeline sin
        pedir nada. Con `monitor.auto_advance: false` solo observa.
        """
        return bool(self.cfg.get("auto_advance", True))

    def _orchestrator_vivo(self) -> bool:
        """Si el orquestador lanzado sigue corriendo.

        La bandera sola no basta: se ponía a `True` al lanzar y solo se limpiaba
        cuando aparecía un batch preparado. Si el orquestador moría —por ejemplo
        con el volumen de `runs/` desmontado— la bandera quedaba fija y el
        monitor se creía ocupado para siempre: ni reintentaba ni avisaba. Ahora
        el fallo se cosecha y se reporta.
        """
        if not getattr(self, "_orchestrator_running", False):
            return False

        proc = getattr(self, "_orchestrator_proc", None)
        if proc is None:
            return True   # lanzado por una versión anterior; no lo damos por muerto

        codigo = proc.poll()
        if codigo is None:
            return True

        self._orchestrator_running = False
        self._orchestrator_proc = None

        fh = getattr(self, "_orchestrator_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._orchestrator_log_fh = None

        if codigo == 0:
            self._orchestrator_fallos = 0
            self._orchestrator_error = None
            return False

        detalle = ""
        try:
            lineas = [ln for ln in orchestrator_log().read_text(
                encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            detalle = lineas[-1] if lineas else ""
        except Exception:
            pass

        self._orchestrator_fallos = getattr(self, "_orchestrator_fallos", 0) + 1
        self._orchestrator_ultimo_fallo = time.time()
        self._orchestrator_error = detalle or f"código {codigo}"
        log.error(
            "El orquestador terminó con código %s (fallo %d)%s",
            codigo, self._orchestrator_fallos, f" — {detalle}" if detalle else "",
        )
        return False

    def _orchestrator_en_espera(self) -> bool:
        """Si toca esperar antes de reintentar tras un fallo.

        Cosechar el fallo sin más convierte el bloqueo permanente en un bucle de
        crashes: cada ciclo relanzaría el mismo orquestador contra la misma
        causa. El backoff exponencial deja que una condición transitoria —montar
        el volumen, liberar disco— se resuelva sola sin inundar el log, y se
        rinde tras varios intentos en vez de insistir eternamente.
        """
        fallos = getattr(self, "_orchestrator_fallos", 0)
        if fallos == 0:
            return False
        if fallos >= ORCHESTRATOR_MAX_FALLOS:
            if not getattr(self, "_orchestrator_rendido", False):
                self._orchestrator_rendido = True
                log.error(
                    "Orquestador desactivado tras %d fallos seguidos: %s. "
                    "Corrige la causa y reinicia el monitor.",
                    fallos, getattr(self, "_orchestrator_error", "?"),
                )
            return True

        espera = min(ORCHESTRATOR_BACKOFF_S * 2 ** (fallos - 1), ORCHESTRATOR_BACKOFF_MAX_S)
        transcurrido = time.time() - getattr(self, "_orchestrator_ultimo_fallo", 0.0)
        return transcurrido < espera

    def _auto_revive_runner(self) -> bool:
        """Si el monitor puede relanzar runners caídos con jobs pendientes."""
        return bool(self.cfg.get("auto_revive_runner", self._auto_advance()))

    @staticmethod
    def _batch_id_from_dir(batch_dir: Path) -> int:
        digits = "".join(filter(str.isdigit, batch_dir.name))
        return int(digits or "0")

    def _launch_runner(self, batch_dir: Path) -> None:
        """Lanza el runner configurado para el batch actual."""
        root = paths.data_root()
        interprete = platform_caps.runner_python(self.cfg)
        if interprete is None:
            # Congelado, sys.executable es el propio binario: lanzarlo
            # arrancaría otro monitor en bucle en vez del runner.
            log.error(
                "No hay intérprete de Python para lanzar el runner. "
                "Define monitor.python_executable en la configuración."
            )
            return
        slots = self.cfg.get("runner_slots", 2)
        cores = self.cfg.get("runner_cores", 8)
        kind = self._runner_kind()
        salida = _salida_desligada(batch_dir)
        if kind == "phase2_force":
            runner = root / "scripts" / "phase2_force_runner.py"
            batch_id = self._batch_id_from_dir(batch_dir)
            runs_dir = Path(self.cfg.get("phase2_runs_dir", batch_dir.parent))
            if not runs_dir.is_absolute():
                runs_dir = (root / runs_dir).resolve()
            env = os.environ.copy()
            env.setdefault("DFT_RUNS_ROOT", str(runs_dir.parent))
            env["PHASE2_FORCE_RUNS_DIR"] = str(runs_dir)
            subprocess.Popen(
                [
                    interprete,
                    str(runner),
                    "--batch-id",
                    str(batch_id),
                    "--slots",
                    str(slots),
                    "--cores",
                    str(cores),
                    "--resume",
                    "--start-real",
                    "--runs-dir",
                    str(runs_dir),
                ],
                cwd=str(root),
                env=env,
                stdout=salida,
                stderr=subprocess.STDOUT,
                **_grupo_propio(),
            )
            return

        runner = root / "scripts" / "buho_relax_runner.py"
        subprocess.Popen(
            [
                interprete,
                str(runner),
                "--relax-dir",
                str(batch_dir),
                "--slots",
                str(slots),
                "--cores",
                str(cores),
            ],
            cwd=str(root),
            stdout=salida,
            stderr=subprocess.STDOUT,
            **_grupo_propio(),
        )

    def _runner_running_for(self, batch_dir: Path) -> bool:
        target_dir = batch_dir.resolve()
        kind = self._runner_kind()
        batch_id = self._batch_id_from_dir(batch_dir)
        runs_dir = Path(self.cfg.get("phase2_runs_dir", batch_dir.parent)).resolve()

        def runner_matches(cmd: list[str]) -> bool:
            if kind == "phase2_force":
                if not any("phase2_force_runner.py" in part for part in cmd):
                    return False
                joined = " ".join(cmd)
                if str(target_dir) in joined:
                    return True
                if "--batch-id" in cmd:
                    try:
                        if int(cmd[cmd.index("--batch-id") + 1]) != batch_id:
                            return False
                    except (ValueError, IndexError):
                        return False
                else:
                    return False
                if "--runs-dir" in cmd:
                    try:
                        proc_runs = Path(cmd[cmd.index("--runs-dir") + 1]).resolve()
                    except (ValueError, IndexError):
                        return False
                    return proc_runs == runs_dir
                return str(runs_dir) in joined

            if not any("buho_relax_runner.py" in part for part in cmd):
                return False
            try:
                proc_dir = Path(cmd[cmd.index("--relax-dir") + 1]).resolve()
            except (ValueError, IndexError):
                return str(target_dir) in " ".join(cmd)
            return proc_dir == target_dir

        return any(
            runner_matches(p.info.get("cmdline") or [])
            for p in __import__("psutil").process_iter(["cmdline"])
        )

    async def poll_once(self) -> None:
        # Se marca siempre, aunque runs_dir no exista: permite a /api/health
        # distinguir "el poller vive pero el volumen está caído" de "el poller
        # está muerto".
        self.last_poll_at = time.time()
        if not self.runs_dir.exists():
            return

        for job_dir in sorted(self.runs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            try:
                snap = snapshot_job(job_dir, self.cfg)
                self._snapshots[snap.job_id] = snap
                await self._check_transition(snap)
            except Exception as exc:
                # WARNING y no DEBUG: un job que falla al parsearse desaparece
                # del monitor entero (API, resumen y Telegram) y en DEBUG nadie
                # se enteraba. Así fue como 203 jobs `skipped_duplicate` se
                # volvieron invisibles.
                log.warning("Job %s ignorado, no se pudo parsear: %s", job_dir.name, exc)

    async def _check_transition(self, snap: StatsResponse) -> None:
        prev = self._prev_states.get(snap.job_id)
        current = snap.status

        if prev == current:
            return

        self._prev_states[snap.job_id] = current

        # Solo eventos que requieren atención inmediata
        event_map = {
            "failed":      "FAILED",
            "stopped":     "STOPPED",
            "stalled":     "STALLED",
            "oscillating": "OSCILLATING",
        }
        event_name = event_map.get(current)

        # WebSocket siempre recibe todos los cambios de estado
        all_events = {
            "running": "STARTED", "converged": "CONVERGED",
            **event_map,
        }
        ws_event_name = all_events.get(current)
        if ws_event_name is None:
            return

        data: dict[str, Any] = {
            "formula":      snap.formula,
            "elapsed_min":  snap.elapsed_min,
            "energy_ev":    snap.energy_history[-1] if snap.energy_history else None,
            "n_fire_steps": snap.n_fire_steps,
            "n_scf_iters":  snap.n_scf_iters,
            "mpi_cores":    snap.mpi_cores,
        }
        if event_name == "OSCILLATING":
            data["energy_history"] = snap.energy_history[-10:]
            data["energy_std"] = (
                round(statistics.stdev(snap.energy_history[-10:]), 4)
                if len(snap.energy_history) >= 2 else None
            )
        if event_name == "STALLED":
            data["stall_minutes"] = snap.stall_minutes

        ws_event = WsEvent(
            job_id=snap.job_id,
            event=ws_event_name,  # type: ignore[arg-type]
            timestamp=datetime.now().isoformat(),
            data=data,
        )

        # WebSocket recibe todo
        try:
            self._event_queue.put_nowait(ws_event)
        except asyncio.QueueFull:
            pass

        # Telegram solo para problemas
        if event_name is not None:
            token   = _telegram_bot_token(self.telegram)
            chat_id = _telegram_chat_id(self.telegram)
            if token and chat_id:
                await send_telegram(token, chat_id, event_name, snap.job_id, data)

    def _find_next_batch(self) -> Path | None:
        """Busca en batches_dir el primer batch con jobs pendientes."""
        batches_dir = paths.resolve_data(
            self.cfg.get("batches_dir", "runs/batches")
        ).resolve()
        if not batches_dir.exists():
            return None
        # Ordenar batch_NNN numéricamente
        candidates = sorted(
            (d for d in batches_dir.iterdir() if d.is_dir()),
            key=lambda d: int("".join(filter(str.isdigit, d.name)) or "0"),
        )
        for batch_dir in candidates:
            # Saltar batches ya lanzados por el monitor
            if (batch_dir / ".runner_launched").exists():
                continue
            has_pending = any(
                json.loads((j / "status.json").read_text()).get("status") == "pending"
                for j in batch_dir.iterdir()
                if j.is_dir() and (j / "status.json").exists()
            )
            if has_pending:
                return batch_dir
        return None

    async def _check_batch_done(self) -> None:
        """Detecta fin de batch, notifica y lanza el siguiente si existe."""
        if not self._snapshots:
            return
        pending = sum(1 for s in self._snapshots.values() if s.status == "pending")
        running = sum(1 for s in self._snapshots.values()
                      if s.status in ("running", "stalled", "oscillating"))
        if pending + running > 0:
            self._batch_done_notified = False
            return
        if getattr(self, "_batch_done_notified", False):
            return

        self._batch_done_notified = True
        n_conv = sum(1 for s in self._snapshots.values() if s.status == "converged")
        n_fail = sum(1 for s in self._snapshots.values() if s.status in ("failed", "stopped"))
        token   = _telegram_bot_token(self.telegram)
        chat_id = _telegram_chat_id(self.telegram)

        next_batch = self._find_next_batch() if self._auto_advance() else None
        if not self._auto_advance():
            log.info("Lote terminado; auto_advance desactivado — no se avanza nada.")

        if next_batch:
            slots = self.cfg.get("runner_slots", 2)
            cores = self.cfg.get("runner_cores", 22)
            self._launch_runner(next_batch)
            (next_batch / ".runner_launched").write_text(
                datetime.now().isoformat()
            )
            log.info("Nuevo batch lanzado: %s (%s slots × %s cores)", next_batch.name, slots, cores)
            # Cambiar el poller al nuevo directorio
            self.runs_dir = next_batch
            self._snapshots.clear()
            self._prev_states.clear()
            self._batch_done_notified = False
            self._orchestrator_running = False   # batch preparado consumido

            if token and chat_id:
                text = (
                    f"✅ <b>Batch completo</b> — {n_conv} conv / {n_fail} fallidos\n"
                    f"🚀 Lanzando <b>{next_batch.name}</b> ({slots} slots × {cores} cores)"
                )
                await send_telegram(token, chat_id, "PING", "batch-done", {"_raw": text})
        else:
            # No hay batch preparado en cola → disparar el orquestador de active
            # learning: finaliza el batch actual (acumula + reentrena + evalúa +
            # grafica) y genera+prepara el siguiente (sin lanzar). El monitor lo
            # lanzará en un poll posterior al detectar el batch preparado.
            al_state_f = paths.resolve_data("data/batches/orchestrator_state.json")
            try:
                al_state = json.loads(al_state_f.read_text()) if al_state_f.exists() else {}
            except Exception:
                al_state = {}

            if al_state.get("stopped"):
                log.info("Active learning terminado: %s", al_state.get("reason"))
                if token and chat_id and not getattr(self, "_al_done_notified", False):
                    self._al_done_notified = True
                    await send_telegram(token, chat_id, "PING", "al-done",
                        {"_raw": f"🏁 <b>Active learning terminado</b>\n{al_state.get('reason','')}"})
            elif not self._auto_advance():
                log.info("auto_advance desactivado — no se dispara el orquestador.")
            elif self._orchestrator_vivo():
                # Ya está trabajando; seguir re-chequeando sin re-spawn.
                self._batch_done_notified = False
            elif self._orchestrator_en_espera():
                self._batch_done_notified = False
            else:
                raiz_datos = paths.data_root()
                orch = raiz_datos / "scripts" / "active_learning_orchestrator.py"
                if not orch.is_file():
                    log.error("No se dispara el orquestador: falta %s", orch)
                    return
                interprete = platform_caps.runner_python(self.cfg)
                if interprete is None:
                    log.error("Sin intérprete de Python: no se dispara el orquestador.")
                    return
                # stderr va a un archivo, no a `subprocess.PIPE`: una tubería
                # sin drenar se llena a los 64 KB y bloquea al orquestador para
                # siempre. Además deja registro consultable del fallo.
                log_orch = orchestrator_log()
                log_orch.parent.mkdir(parents=True, exist_ok=True)
                self._orchestrator_log_fh = open(log_orch, "a", encoding="utf-8")
                self._orchestrator_log_fh.write(
                    f"\n===== {datetime.now().isoformat(timespec='seconds')} advance =====\n"
                )
                self._orchestrator_log_fh.flush()
                try:
                    self._orchestrator_proc = subprocess.Popen(
                        [interprete, str(orch), "advance"], cwd=str(raiz_datos),
                        # stdout también al archivo: heredarlo del motor ataba
                        # el orquestador a la vida de la app, que lo lee por
                        # tubería. Al cerrarla moría de SIGPIPE.
                        stdout=self._orchestrator_log_fh,
                        stderr=subprocess.STDOUT, text=True,
                        **_grupo_propio(),
                    )
                except OSError as exc:
                    # Sin esto el descriptor recién abierto se fuga en cada
                    # ciclo del poller mientras la causa persista.
                    self._orchestrator_log_fh.close()
                    self._orchestrator_log_fh = None
                    self._orchestrator_fallos = getattr(self, "_orchestrator_fallos", 0) + 1
                    self._orchestrator_ultimo_fallo = time.time()
                    self._orchestrator_error = str(exc)
                    log.error("No se pudo lanzar el orquestador: %s", exc)
                    return
                self._orchestrator_running = True
                self._batch_done_notified = False   # re-chequear hasta que aparezca el batch
                log.info("Orquestador disparado (retrain + preparar siguiente batch)")
                if token and chat_id:
                    await send_telegram(token, chat_id, "PING", "al-prep",
                        {"_raw": f"✅ <b>Batch completo</b> — {n_conv} conv / {n_fail} fallidos\n"
                                 f"🧠 Reentrenando surrogate + preparando siguiente batch…"})

    async def _check_runner_alive(self) -> None:
        """Detecta runner muerto (pending>0, running==0) y lo relanza automáticamente."""
        if not self._auto_revive_runner():
            self._runner_dead_since = 0.0
            return
        if not self._snapshots:
            return
        pending = sum(1 for s in self._snapshots.values() if s.status == "pending")
        running = sum(1 for s in self._snapshots.values()
                      if s.status in ("running", "stalled", "oscillating"))
        if pending == 0 or running > 0:
            self._runner_dead_since = 0.0
            return

        # pending > 0 y running == 0 — ¿hace cuánto?
        now = time.time()
        if not getattr(self, "_runner_dead_since", 0.0):
            self._runner_dead_since = now
            return

        # Esperar 2 ciclos completos antes de actuar (evitar falsos positivos al arrancar)
        wait_sec = self.cfg.get("poll_interval_sec", 30) * 2
        if now - self._runner_dead_since < wait_sec:
            return

        if self._runner_running_for(self.runs_dir):
            self._runner_dead_since = 0.0
            return

        # Runner realmente muerto → relanzar
        self._runner_dead_since = 0.0
        slots = self.cfg.get("runner_slots", 2)
        cores = self.cfg.get("runner_cores", 8)
        self._launch_runner(self.runs_dir)
        log.warning("Runner muerto detectado — relanzando para %s", self.runs_dir.name)
        token   = _telegram_bot_token(self.telegram)
        chat_id = _telegram_chat_id(self.telegram)
        if token and chat_id:
            await send_telegram(token, chat_id, "PING", "runner-revived", {
                "_raw": (
                    f"⚠️ <b>Runner relanzado</b>\n"
                    f"Batch: <code>{self.runs_dir.name}</code>\n"
                    f"{pending} jobs pendientes  {slots} slots × {cores} cores"
                )
            })

    async def _check_phase2_no_scf_self_heal(self) -> None:
        """Mata jobs Fase 2A vivos que no producen ninguna iteracion SCF."""
        if self._runner_kind() != "phase2_force":
            return
        threshold = float(self.cfg.get("phase2_no_scf_stall_minutes", 60))
        if threshold <= 0 or not self.runs_dir.exists():
            return
        token = self.telegram.get("bot_token", "")
        chat_id = _telegram_chat_id(self.telegram)

        for job_dir in sorted(self.runs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            st = _read_status(job_dir)
            if st.get("status") != "running":
                continue
            pid = st.get("pid")
            if isinstance(pid, int) and not _is_pid_alive(pid):
                continue
            started_epoch = _parse_epoch(st.get("started_at") or st.get("start_time"))
            if started_epoch is None:
                continue
            elapsed_min = (time.time() - started_epoch) / 60
            if elapsed_min < threshold:
                continue

            progress = _phase2_scf_progress(job_dir, st)
            newest = progress.get("newest")
            reason = None
            if newest and int(newest.get("last_iter") or 0) == 0 and float(newest.get("age_min") or 0) >= threshold:
                reason = (
                    f"no_scf_iterations_after_{elapsed_min:.1f}min; "
                    f"latest_log={newest['relative_path']}; "
                    f"log_age={float(newest['age_min']):.1f}min; "
                    f"log_size={newest['size']}"
                )
            elif not newest:
                reason = f"no_phase2_scf_log_after_{elapsed_min:.1f}min"
            if not reason:
                continue

            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _write_status_update(job_dir, {
                "status": "failed",
                "self_healed": True,
                "self_heal_action": "monitor_killed_stalled_no_scf_job",
                "self_heal_reason": reason,
                "finished_at": now,
                "updated_at": now,
            })
            killed = _kill_job_processes(job_dir, pid if isinstance(pid, int) else None)
            _write_status_update(job_dir, {"self_heal_pids": killed})
            log.warning("Self-healer mato job sin SCF: %s reason=%s", job_dir.name, reason)
            if token and chat_id:
                await send_telegram(token, chat_id, "PING", "phase2-self-heal", {
                    "_raw": (
                        "🩺 <b>Self-healer Fase 2A</b>\n"
                        f"Job: <code>{job_dir.name}</code>\n"
                        f"Formula: <code>{st.get('formula', job_dir.name)}</code>\n"
                        f"Motivo: <code>{reason}</code>\n"
                        "Accion: marcado failed y procesos terminados."
                    )
                })

    async def _check_phase2_u_oscillation_self_heal(self) -> None:
        """Mata jobs Fase 2A con oscilacion U-scan y acepta la U previa si existe."""
        if self._runner_kind() != "phase2_force":
            return
        if not self.cfg.get("phase2_u_oscillation_self_heal", True):
            return
        if not self.runs_dir.exists():
            return
        token = self.telegram.get("bot_token", "")
        chat_id = _telegram_chat_id(self.telegram)

        for job_dir in sorted(self.runs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            st = _read_status(job_dir)
            if st.get("status") != "running":
                continue
            pid = st.get("pid")
            if isinstance(pid, int) and not _is_pid_alive(pid):
                continue

            decision = evaluate_u_oscillation_self_heal(
                job_dir,
                st,
                self.cfg,
                started_epoch=_parse_epoch(st.get("started_at") or st.get("start_time")),
            )
            if not decision:
                continue

            _write_status_update(job_dir, status_update_for_u_decision(decision, []))
            killed = _kill_job_processes(job_dir, pid if isinstance(pid, int) else None)
            write_u_scan_decision(job_dir, decision, killed)
            _write_status_update(job_dir, status_update_for_u_decision(decision, killed))
            log.warning(
                "Self-healer U-scan mato job: %s action=%s accepted_u=%s rejected_u=%s reason=%s",
                job_dir.name,
                decision["self_heal_action"],
                decision.get("accepted_u_ev"),
                decision.get("rejected_u_ev"),
                decision["reason"],
            )
            if token and chat_id:
                accepted = decision.get("accepted_u_ev")
                accepted_text = (
                    f"U aceptada: <code>{accepted}</code> eV\n"
                    if accepted is not None
                    else "U aceptada: <code>ninguna</code>\n"
                )
                await send_telegram(token, chat_id, "PING", "phase2-u-self-heal", {
                    "_raw": (
                        "🩺 <b>Self-healer U-scan Fase 2A</b>\n"
                        f"Job: <code>{job_dir.name}</code>\n"
                        f"Formula: <code>{st.get('formula', job_dir.name)}</code>\n"
                        f"U rechazada: <code>{decision.get('rejected_u_ev')}</code> eV\n"
                        f"{accepted_text}"
                        f"Motivo: <code>{decision['reason']}</code>\n"
                        f"Accion: <code>{decision['self_heal_action']}</code>"
                    )
                })

    async def _check_surrogate(self) -> None:
        """Notifica cuando el surrogate se reentrenó (metrics.json más nuevo)."""
        rel = self.cfg.get("surrogate_metrics_path",
                           "models/surrogate_energy.pkl.metrics.json")
        metrics_path = paths.resolve_data(rel).resolve()
        if not metrics_path.exists():
            return
        mtime = metrics_path.stat().st_mtime
        if mtime <= getattr(self, "_last_surrogate_mtime", 0):
            return
        self._last_surrogate_mtime = mtime
        token   = _telegram_bot_token(self.telegram)
        chat_id = _telegram_chat_id(self.telegram)
        if not token or not chat_id:
            return
        try:
            m = json.loads(metrics_path.read_text())
            lines = ["🤖 <b>Surrogate entrenado</b>"]
            if "n_train" in m:
                lines.append(f"n_train: {m['n_train']}   n_features: {m.get('n_features','?')}")
            if "cv_mae" in m:
                lines.append(f"CV MAE: {m['cv_mae']:.4f} eV/átomo")
            if "train_mae" in m:
                lines.append(f"Train MAE: {m['train_mae']:.4f} eV/átomo")
            if "test_mae" in m:
                lines.append(f"Test MAE: {m['test_mae']:.4f} eV/átomo")
            if "overfit_ratio" in m:
                lines.append(f"Overfit ratio: {m['overfit_ratio']:.2f}")
            await send_telegram(token, chat_id, "PING", "surrogate", {"_raw": "\n".join(lines)})
        except Exception as exc:
            log.error("Error leyendo metrics surrogate: %s", exc)

    async def _check_temperatures(self) -> None:
        """Envía alerta si algún socket CPU supera el umbral configurado."""
        from .sysmetrics import collect as collect_metrics
        threshold = self.cfg.get("temp_alert_celsius", 85)
        token   = _telegram_bot_token(self.telegram)
        chat_id = _telegram_chat_id(self.telegram)
        if not token or not chat_id:
            return
        try:
            m = collect_metrics()
            hot = [(i, t) for i, t in enumerate(m.pkg_temps) if t >= threshold]
            if not hot:
                self._temp_alerted = False
                return
            if getattr(self, "_temp_alerted", False):
                return  # ya se notificó, no spam
            self._temp_alerted = True
            pkg_str = "  ".join(f"PKG{i}: {t}°C" for i, t in hot)
            text = (
                f"🌡️ <b>TEMPERATURA ALTA</b>\n"
                f"{pkg_str}\n"
                f"núcleo max: {m.core_temp_max}°C   CPU: {m.cpu_percent:.0f}%"
            )
            await send_telegram(token, chat_id, "PING", "temp-alert", {"_raw": text})
        except Exception as exc:
            log.error("Error en check_temperatures: %s", exc)

    def _raiz_de_lotes(self) -> Path:
        """Directorio que contiene los `batch_NNN`.

        Se prefiere lo configurado, pero **solo si existe**: el default
        histórico era `runs/batches`, que en esta instalación no existe —los
        lotes viven en `local_runs/phase2_force/`— y bastaba para que la
        búsqueda del lote activo devolviera None siempre.
        """
        for clave in ("phase2_runs_dir", "batches_dir"):
            configurada = self.cfg.get(clave) or ""
            if configurada:
                raiz = paths.resolve_data(configurada)
                if raiz.is_dir():
                    return raiz.resolve()

        # Si `runs_dir` ya es un batch_NNN, sus hermanos son los demás lotes.
        if _BATCH_RE.match(self.runs_dir.name):
            return self.runs_dir.parent.resolve()
        return self.runs_dir.resolve()

    @staticmethod
    def _tiene_activos(batch_dir: Path) -> bool:
        """Si al lote le quedan jobs por hacer."""
        try:
            for job in batch_dir.iterdir():
                if not job.is_dir():
                    continue
                st = job / "status.json"
                if not st.is_file():
                    continue
                try:
                    estado = json.loads(st.read_text(encoding="utf-8")).get("status")
                except (OSError, json.JSONDecodeError):
                    continue
                if estado in ("pending", "running", "stalled", "oscillating"):
                    return True
        except OSError:
            return False
        return False

    def _find_active_batch(self) -> Path | None:
        """El lote más reciente con trabajo pendiente.

        Ya no se exige el centinela `.runner_launched`: ningún lote preparado
        desde el cribado lo escribe, así que era un filtro que descartaba
        justamente los lotes que interesaba encontrar. Lo que define un lote
        activo es tener jobs sin terminar.
        """
        raiz = self._raiz_de_lotes()
        if not raiz.is_dir():
            return None

        lotes = sorted(
            (d for d in raiz.iterdir() if d.is_dir() and _BATCH_RE.match(d.name)),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for batch_dir in lotes:
            if self._tiene_activos(batch_dir):
                return batch_dir
        return None

    def _seguir_lote_activo(self) -> None:
        """Mueve `runs_dir` al lote que de verdad tiene trabajo.

        Esto corría **una sola vez, al arrancar el poller**. En cuanto se
        lanzaba un lote nuevo, el monitor seguía mirando el anterior: la franja
        de actividad decía «en reposo» con veinte procesos GPAW calculando, los
        candidatos eran siempre los del mismo lote, el visor enseñaba
        estructuras de hace días y el panel de logs no encontraba nada. Ahora se
        reevalúa en cada ciclo, que cuesta un `iterdir` y para en el primer job
        pendiente que encuentra.
        """
        if self._tiene_activos(self.runs_dir):
            return          # el lote vigilado sigue vivo: no hay nada que mover

        activo = self._find_active_batch()
        if activo is None or activo.resolve() == self.runs_dir.resolve():
            return

        # Antes iba dentro de un `except Exception: pass`, así que un fallo aquí
        # desaparecía sin rastro y el síntoma era «el monitor no ve mi lote».
        log.info("Siguiendo el lote activo: %s -> %s",
                 self.runs_dir.name, activo.name)
        self.runs_dir = activo
        self._batch_done_notified = False
        self._snapshots.clear()

    async def run_forever(self, interval: int) -> None:
        log.info("Poller iniciado — directorio=%s  intervalo=%ss", self.runs_dir, interval)
        self._temp_alerted = False
        self._batch_done_notified = False
        self._last_surrogate_mtime = 0.0

        self._seguir_lote_activo()

        while True:
            # Antes de cada sondeo, no solo al arrancar: un lote lanzado después
            # nunca se detectaba.
            self._seguir_lote_activo()
            try:
                await self.poll_once()
            except Exception as exc:
                log.error("Error en poll_once: %s", exc)

            await self._check_temperatures()
            try:
                await self._check_runner_alive()
            except Exception as exc:
                log.error("Error en _check_runner_alive: %s", exc)
            try:
                await self._check_phase2_no_scf_self_heal()
            except Exception as exc:
                log.error("Error en _check_phase2_no_scf_self_heal: %s", exc)
            try:
                await self._check_phase2_u_oscillation_self_heal()
            except Exception as exc:
                log.error("Error en _check_phase2_u_oscillation_self_heal: %s", exc)
            try:
                await self._check_batch_done()
            except Exception as exc:
                log.error("Error en _check_batch_done: %s", exc)
            try:
                await self._check_surrogate()
            except Exception as exc:
                log.error("Error en _check_surrogate: %s", exc)
            await asyncio.sleep(interval)
