"""Wizard de entorno para el monitor.

Expone `buho.setup_wizard` por HTTP: la GUI enseña qué falta y lanza la
instalación sin que nadie tenga que abrir una terminal ni saber si el paquete
va a Windows o a WSL.

La instalación corre en un hilo con el log en memoria, no como subproceso
desacoplado (que es lo que hace `bench`): un `pip install` dura minutos y el
usuario lo está mirando, así que lo que importa es transmitir la salida, no
sobrevivir al cierre de la app.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import yaml

from .. import paths

log = logging.getLogger(__name__)

#: Cuántas líneas de salida se guardan. pip es verboso y la GUI solo enseña
#: la cola; retener el log entero de un torch sería decenas de MB por instalación.
MAX_LINEAS = 2000

_lock = threading.Lock()
_thread: threading.Thread | None = None
_job: dict[str, Any] = {}
_log: deque[str] = deque(maxlen=MAX_LINEAS)


def _config() -> dict[str, Any]:
    ruta = paths.resolve_data("config/generator.yaml")
    if not ruta.is_file():
        ruta = paths.bundle_file("config", "generator.yaml")
    try:
        with ruta.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except OSError:
        return {}


def _corriendo() -> bool:
    return _thread is not None and _thread.is_alive()


def status(*, fast: bool = False) -> dict[str, Any]:
    """Matriz de capacidades, más el estado del trabajo en curso."""
    from buho import setup_wizard

    data = setup_wizard.check(_config(), project_root=paths.data_root(), incluir_mlff=not fast)
    data["job"] = job()
    return data


def job() -> dict[str, Any]:
    """Estado del trabajo de instalación, con la cola del log."""
    with _lock:
        actual = dict(_job)
        actual["running"] = _corriendo()
        actual["log"] = list(_log)
    return actual


def plan(target: str, **opciones: Any) -> dict[str, Any]:
    from buho import setup_wizard

    return setup_wizard.plan(target, config=_config(), **opciones).as_dict()


def start_install(target: str, **opciones: Any) -> dict[str, Any]:
    """Lanza el plan en segundo plano. Devuelve el estado inicial del trabajo."""
    from buho import setup_wizard

    global _thread

    with _lock:
        if _corriendo():
            raise RuntimeError("Ya hay una instalación en curso.")

        # Instalar mientras el protocolo criba dejaría a la cascada importando
        # un entorno a medio escribir. Es exactamente el fallo que este wizard
        # existe para evitar, así que no se permite provocarlo desde aquí.
        try:
            from . import discovery as discovery_service

            if discovery_service.status().get("background", {}).get("running"):
                raise RuntimeError(
                    "Pausa el protocolo de descubrimiento antes de instalar dependencias."
                )
        except ImportError:
            pass

        plan_obj = setup_wizard.plan(target, config=_config(), **opciones)
        if not plan_obj.steps:
            return {"status": "skipped", "target": target, "notas": plan_obj.notas,
                    "running": False, "log": []}

        _log.clear()
        _job.clear()
        _job.update({
            "target": target,
            "status": "running",
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
            "plan": plan_obj.as_dict(),
        })

        def _emit(linea: str) -> None:
            with _lock:
                _log.append(linea)

        def _target_fn() -> None:
            try:
                resultado = setup_wizard.execute(plan_obj, on_output=_emit)
            except Exception as exc:  # noqa: BLE001 - un hilo no puede morir mudo
                log.exception("Instalación '%s' falló", target)
                resultado = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            # El entorno recien creado no sirve de nada si nadie sabe donde
            # esta: sin escribir discovery.wsl, `setup_wizard` seguiria pidiendo
            # que se defina a mano y el arranque rapido seguiria negandose.
            if target == "dft" and resultado.get("status") == "ok":
                try:
                    resultado["configuracion"] = configurar_wsl_dft(
                        distro=opciones.get("distro"),
                        env_name=opciones.get("env_name"),
                    )
                except Exception as exc:  # noqa: BLE001 - no puede tumbar el hilo
                    log.exception("no se pudo escribir discovery.wsl")
                    resultado["configuracion"] = {
                        "escrito": False, "motivo": f"{type(exc).__name__}: {exc}"}

            with _lock:
                _job["status"] = resultado.get("status", "error")
                _job["error"] = resultado.get("error")
                _job["steps"] = resultado.get("steps", [])
                _job["configuracion"] = resultado.get("configuracion")
                _job["finished_at"] = time.time()

        _thread = threading.Thread(target=_target_fn, name=f"perovowl-setup-{target}",
                                   daemon=True)
        _thread.start()

    return job()


def reset_for_tests() -> None:
    global _thread
    with _lock:
        _thread = None
        _job.clear()
        _log.clear()


# ── Configuracion de WSL tras instalar el runtime DFT ────────────────────────

def _raiz_datos_en_wsl() -> str | None:
    """La raiz de datos vista desde WSL, p. ej. C:/Users/x -> /mnt/c/Users/x.

    Es la pieza que hace utilizable el entorno recien creado: ahi es donde
    `paths.materializar_pipeline` deja `scripts/` y `src/`, y el runner de DFT
    los importa desde el Python de WSL. Sin esta traduccion el entorno existe
    pero no encuentra el pipeline.
    """
    raiz = paths.data_root()
    partes = raiz.parts
    if not partes or ":" not in partes[0]:
        return None
    unidad = partes[0][0].lower()
    resto = "/".join(partes[1:])
    return f"/mnt/{unidad}/{resto}" if resto else f"/mnt/{unidad}"


def configurar_wsl_dft(*, distro: str | None = None,
                       env_name: str | None = None) -> dict[str, Any]:
    """Escribe `discovery.wsl` en el generator.yaml del usuario.

    Se llama al terminar bien la instalacion del runtime DFT. Sin esto el
    entorno queda creado pero nadie sabe donde esta: `setup_wizard` seguiria
    diciendo «Define discovery.wsl.python».
    """
    from buho import setup_wizard

    distros = setup_wizard.distros_wsl()
    if not distros:
        return {"escrito": False, "motivo": "no hay distros WSL"}
    distro = distro or distros[0]
    env = env_name or setup_wizard.GPAW_ENV
    rutas = setup_wizard._rutas_gpaw(_config(), env)

    destino = paths.resolve_data("config/generator.yaml")
    cfg: dict[str, Any] = {}
    if destino.is_file():
        try:
            with destino.open(encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            return {"escrito": False, "motivo": f"no se pudo leer {destino}: {exc}"}
    else:
        # Instalacion nueva: se parte de la copia empaquetada, no de cero, para
        # no perder el espacio quimico ni las puertas del cribado.
        try:
            with paths.bundle_file("config", "generator.yaml").open(encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            cfg = {}

    discovery = cfg.setdefault("discovery", {})
    if not isinstance(discovery, dict):
        discovery = cfg["discovery"] = {}
    wsl = discovery.setdefault("wsl", {})
    if not isinstance(wsl, dict):
        wsl = discovery["wsl"] = {}

    proyecto = _raiz_datos_en_wsl()
    wsl.update({
        "distro": distro,
        "env_name": env,
        "micromamba": rutas["micromamba"],
        "python": rutas["python"],
        "mpirun": rutas["mpirun"],
        "setup_path": rutas["setups"],
        "driver_python": "python3",
    })
    if proyecto:
        wsl["project_root"] = proyecto

    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        parcial = destino.with_suffix(".yaml.parcial")
        with parcial.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
        parcial.replace(destino)
    except OSError as exc:
        return {"escrito": False, "motivo": f"{type(exc).__name__}: {exc}"}

    return {"escrito": True, "fichero": str(destino), "distro": distro,
            "env": env, "python": rutas["python"], "project_root": proyecto}
