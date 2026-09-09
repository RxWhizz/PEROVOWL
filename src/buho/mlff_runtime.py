"""Donde corre el tier MLFF/GNN del cribado (Tier 2 de la cascada).

Tier 2 necesita `torch` + `matgl` + `pymatgen`: ~2 GB que el interprete del
monitor no usa para nada mas, y que en Windows son la parte mas fragil de la
pila (las ruedas de matgl/dgl). El runtime de GPAW ya vive en WSL, asi que ese
es el sitio natural para Tier 2 tambien -- pero en su **propio** entorno:
`gpaw246` fija numpy 1.26 porque GPAW lo necesita, y matgl arrastra numpy>=2.
Mezclarlos rompe el DFT que ya funciona.

Este modulo es la unica fuente de verdad de esa decision. Lo comparten la
cascada (que necesita ejecutar Tier 2) y `buho setup` (que necesita instalarlo
y explicar por que falta). Nada mas deberia decidir donde corre el MLFF.

Precedencia, de mayor a menor: variables de entorno, `discovery.mlff` en
generator.yaml, deteccion automatica.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Nombre por defecto del entorno WSL dedicado al MLFF. Deliberadamente NO es
#: `gpaw246`: ver el docstring del modulo.
DEFAULT_WSL_ENV = "perovowl-mlff"

#: Candidatos por llamada al worker. Medido en esta maquina: cargar
#: MEGNet+M3GNet cuesta ~15 s y cada candidato ~0.08 s. Con 2000, la carga se
#: amortiza (<1% del tiempo), el JSON por stdin se queda en ~2 MB y un pool de
#: 5000 se resuelve en tres llamadas en vez de en una de siete minutos.
DEFAULT_CHUNK = 2000

#: Tiempo por defecto para la seleccion de fase. Elegir fase no es predecir:
#: cada estructura son cinco relajaciones FIRE sobre celdas de 40 atomos con
#: filtro de celda, y el lote no se trocea. El timeout de prediccion (900 s) se
#: agotaria a mitad y no habria forma de distinguirlo de un entorno roto.
DEFAULT_TIMEOUT_FASES = 3600

#: Paquetes que Tier 2 necesita para importar, en orden de coste de fallo.
REQUIRED_MODULES = ("torch", "matgl", "pymatgen")

MLFF_ENV_KEYS = {
    "backend": ("BUHO_MLFF_BACKEND",),
    "python": ("BUHO_MLFF_PYTHON",),
    "worker": ("BUHO_MLFF_WORKER",),
    "timeout": ("BUHO_MLFF_TIMEOUT",),
    "timeout_fases": ("BUHO_MLFF_TIMEOUT_FASES",),
}
WSL_MLFF_ENV_KEYS = {
    "distro": ("BUHO_WSL_MLFF_DISTRO", "BUHO_WSL_DISTRO"),
    "env_name": ("BUHO_WSL_MLFF_ENV",),
    "micromamba": ("BUHO_WSL_MICROMAMBA",),
    "project_root": ("BUHO_WSL_MLFF_PROJECT_ROOT", "BUHO_WSL_PROJECT_ROOT"),
    "python": ("BUHO_WSL_MLFF_PYTHON",),
}

#: Ruta del worker dentro del repo, relativa a la raiz del proyecto.
WORKER_REL = "scripts/buho_mlff_worker.py"


class MLFFUnavailableError(RuntimeError):
    """Tier 2 no puede ejecutarse aqui.

    Se distingue de un `ModuleNotFoundError` suelto a proposito: la cascada
    tiene que poder degradar a Tier 0/1 sin confundir "falta el entorno MLFF"
    con "hay un bug importando algo". Lleva la remediacion concreta para que la
    GUI y el CLI no la reinventen cada uno por su lado.
    """

    def __init__(self, mensaje: str, *, remediation: str = "") -> None:
        super().__init__(mensaje)
        self.remediation = remediation


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _clean(os.environ.get(name))
        if value:
            return value
    return None


def _windows_path_to_wsl(path: str | Path) -> str:
    """`C:\\Users\\x` -> `/mnt/c/Users/x`.

    Solo la traduccion de unidad. Los montajes personalizados (por ejemplo
    `C:/NuevoVol` -> `/mnt/n`) los resuelve el engine para los runs de DFT; el
    worker MLFF unicamente necesita la raiz del repo, que la config ya declara
    en `discovery.mlff.wsl.project_root` o `discovery.wsl.project_root`.
    """
    raw = str(path).replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":":
        drive = raw[0].lower()
        rest = raw[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    return raw


@dataclass(frozen=True)
class MLFFRuntime:
    """Como invocar el worker MLFF, ya resuelto."""

    backend: str                      # "local" | "wsl" | "off"
    python: str | None = None         # interprete que corre el worker
    worker: str | None = None         # ruta del worker EN EL LADO QUE EJECUTA
    distro: str | None = None         # solo wsl
    project_root: str | None = None   # raiz del repo en el lado que ejecuta
    env_name: str | None = None       # nombre del env micromamba (informativo)
    micromamba: str | None = None     # binario micromamba en WSL (para el wizard)
    timeout: int = 900                # por llamada, no por lote entero
    chunk_size: int = DEFAULT_CHUNK
    timeout_fases: int = DEFAULT_TIMEOUT_FASES   # relajar cuesta otro orden

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "python": self.python,
            "worker": self.worker,
            "distro": self.distro,
            "project_root": self.project_root,
            "env_name": self.env_name,
            "micromamba": self.micromamba,
            "timeout": self.timeout,
            "chunk_size": self.chunk_size,
            "timeout_fases": self.timeout_fases,
        }

    # ── Construccion del comando ──────────────────────────────────────────────

    def command(self, args: list[str]) -> list[str]:
        """Argv completo para lanzar el worker con `args`."""
        if self.backend == "off":
            raise MLFFUnavailableError(
                "El tier MLFF esta desactivado por configuracion.",
                remediation="Pon discovery.mlff.backend en 'auto' (o 'wsl') en config/generator.yaml.",
            )
        if not self.python or not self.worker:
            raise MLFFUnavailableError(
                "El runtime MLFF no tiene interprete ni worker resueltos.",
                remediation="Ejecuta 'buho setup check' para ver que falta.",
            )

        if self.backend == "wsl":
            inner = " ".join(
                shlex.quote(part) for part in [self.python, self.worker, *args]
            )
            if self.project_root:
                inner = f"cd {shlex.quote(self.project_root)} && " + inner
            cmd = ["wsl.exe"]
            if self.distro:
                cmd.extend(["-d", self.distro])
            cmd.extend(["--", "bash", "-lc", inner])
            return cmd

        return [self.python, self.worker, *args]

    # ── Ejecucion ─────────────────────────────────────────────────────────────

    def _run(self, args: list[str], *, stdin: str | None = None, timeout: int | None = None) -> dict[str, Any]:
        cmd = self.command(args)
        try:
            proc = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MLFFUnavailableError(
                f"No se pudo ejecutar el worker MLFF: {exc}",
                remediation=(
                    "Falta wsl.exe en el PATH."
                    if self.backend == "wsl"
                    else "Revisa discovery.mlff.python en config/generator.yaml."
                ),
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MLFFUnavailableError(
                f"El worker MLFF excedio {timeout or self.timeout}s.",
                remediation="Sube discovery.mlff.timeout o reduce discovery.mlff_pool_size.",
            ) from exc

        salida = (proc.stdout or "").strip()
        # El worker imprime UNA linea JSON al final. Cualquier cosa antes es
        # ruido de arranque (avisos de matgl, de WSL) y no debe romper el parseo.
        payload: dict[str, Any] | None = None
        for linea in reversed(salida.splitlines()):
            linea = linea.strip()
            if not linea.startswith("{"):
                continue
            try:
                payload = json.loads(linea)
                break
            except json.JSONDecodeError:
                continue

        if payload is None:
            detalle = (proc.stderr or salida or f"codigo {proc.returncode}").strip()
            # El fallo mas comun con diferencia es que el entorno todavia no
            # existe. Decir "no devolvio JSON" ahi manda a diagnosticar algo que
            # ya se sabe; se traduce al problema real.
            if "No such file or directory" in detalle or "cannot find" in detalle.lower():
                falta_worker = self.worker and self.worker in detalle
                que = "el worker" if falta_worker else "el entorno MLFF"
                raise MLFFUnavailableError(
                    f"No existe {que}: {self.worker if falta_worker else self.python}",
                    remediation=(
                        "Comprueba discovery.mlff.wsl.project_root en config/generator.yaml."
                        if falta_worker
                        else "Ejecuta 'buho setup install mlff' para crearlo."
                    ),
                )
            raise MLFFUnavailableError(
                f"El worker MLFF no devolvio JSON: {detalle[:500]}",
                remediation="Ejecuta 'buho setup check --json' para diagnosticar el entorno.",
            )
        if payload.get("status") == "error":
            raise MLFFUnavailableError(
                payload.get("error") or "El worker MLFF fallo.",
                remediation=payload.get("remediation") or "",
            )
        return payload

    def probe(self, *, timeout: int = 180) -> dict[str, Any]:
        """Comprueba que el worker arranca y que sus modulos importan.

        No lanza: devuelve el diagnostico. El wizard y `/api/setup/status`
        necesitan mostrar el fallo, no propagarlo.
        """
        base = {"backend": self.backend, "python": self.python, "distro": self.distro}
        if self.backend == "off":
            return {**base, "available": False, "error": "MLFF desactivado por configuracion.",
                    "remediation": "Pon discovery.mlff.backend en 'auto' en config/generator.yaml."}
        try:
            payload = self._run(["--preflight-only"], timeout=timeout)
        except MLFFUnavailableError as exc:
            return {**base, "available": False, "error": str(exc), "remediation": exc.remediation}
        return {
            **base,
            "available": True,
            "error": None,
            "remediation": "",
            "versions": payload.get("versions", {}),
            "worker": self.worker,
        }

    def predict(self, candidates: list[dict[str, Any]], config: dict[str, Any], *,
                timeout: int | None = None,
                on_progress: Any = None) -> list[dict[str, Any]]:
        """Evalua Tier 2 sobre un lote de candidatos serializados.

        Se trocea en bloques de `chunk_size`. El tamano es un compromiso medido:
        cargar MEGNet+M3GNet cuesta ~15 s y predecir ~0.08 s por candidato, asi
        que trozos pequenos pagan la carga muchas veces y uno solo de 5000
        significa 5 MB de JSON por stdin, siete minutos sin dar senales y
        perderlo todo si el ultimo candidato falla. Con el valor por defecto,
        un pool de 5000 son tres llamadas.
        """
        if not candidates:
            return []

        tamano = max(1, int(self.chunk_size))
        resultados: list[dict[str, Any]] = []
        for inicio in range(0, len(candidates), tamano):
            trozo = candidates[inicio:inicio + tamano]
            payload = json.dumps({"candidates": trozo, "config": config}, ensure_ascii=False)
            out = self._run(["--stdin"], stdin=payload, timeout=timeout)
            resultados.extend(out.get("results", []))
            if on_progress is not None:
                on_progress(len(resultados), len(candidates))
        return resultados


    def fases(self, estructuras: list[dict[str, Any]], *,
              opciones: dict[str, Any] | None = None,
              timeout: int | None = None) -> list[dict[str, Any]]:
        """Elige la fase de menor energia de cada estructura del lote.

        Por que esto vive detras del worker y no en el motor: `seleccionar_fase`
        relaja con FIRE sobre un filtro de celda, y eso necesita un calculador
        ASE **en proceso** con energia, fuerzas y tension. El potencial esta en
        el entorno del MLFF --- en Windows, dentro de WSL--- y no se puede pasar
        por una tuberia. Lo que cruza es la geometria ganadora.

        El lote NO se trocea. `predict` lo hace porque son miles de candidatos
        de coste uniforme; aqui son las decenas que van a DFT en una ronda, y
        partirlas solo multiplicaria la carga del potencial.
        """
        if not estructuras:
            return []

        payload = json.dumps(
            {"estructuras": estructuras, "opciones": opciones or {}},
            ensure_ascii=False)
        out = self._run(["--fases"], stdin=payload,
                        timeout=timeout or self.timeout_fases)
        return out.get("resultados", [])


# ── Resolucion ────────────────────────────────────────────────────────────────


def _mlff_config(config: dict[str, Any] | None) -> dict[str, Any]:
    discovery = (config or {}).get("discovery", {}) or {}
    return discovery.get("mlff", {}) or {}


def _option(key: str, mlff_cfg: dict[str, Any], *, wsl: bool) -> str | None:
    """Env var > bloque wsl de mlff > bloque plano de mlff."""
    env_keys = WSL_MLFF_ENV_KEYS.get(key) if wsl else MLFF_ENV_KEYS.get(key)
    if env_keys:
        value = _first_env(env_keys)
        if value:
            return value
    if wsl:
        wsl_cfg = mlff_cfg.get("wsl", {}) or {}
        value = _clean(wsl_cfg.get(key))
        if value:
            return value
    return _clean(mlff_cfg.get(key))


def _local_modules_present() -> bool:
    import importlib.util

    for mod in REQUIRED_MODULES:
        try:
            if importlib.util.find_spec(mod) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def resolve(config: dict[str, Any] | None = None, *, project_root: Path | str | None = None) -> MLFFRuntime:
    """Decide donde corre Tier 2 sin ejecutarlo todavia.

    Barata a proposito: solo mira config, entorno y la existencia de rutas. La
    comprobacion de verdad (importar torch) la hace `MLFFRuntime.probe()`, que
    cuesta segundos.
    """
    mlff_cfg = _mlff_config(config)
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]

    backend = (_option("backend", mlff_cfg, wsl=False) or "auto").lower()
    if backend not in {"auto", "local", "wsl", "off"}:
        raise MLFFUnavailableError(
            f"Backend MLFF no reconocido: {backend}",
            remediation="Valores validos: auto, local, wsl, off.",
        )

    try:
        timeout = int(_option("timeout", mlff_cfg, wsl=False) or 900)
    except (TypeError, ValueError):
        timeout = 900
    try:
        chunk = int(_option("chunk_size", mlff_cfg, wsl=False) or DEFAULT_CHUNK)
    except (TypeError, ValueError):
        chunk = DEFAULT_CHUNK
    chunk = max(1, chunk)
    try:
        timeout_fases = int(
            _option("timeout_fases", mlff_cfg, wsl=False) or DEFAULT_TIMEOUT_FASES)
    except (TypeError, ValueError):
        timeout_fases = DEFAULT_TIMEOUT_FASES

    wsl_python = _option("python", mlff_cfg, wsl=True)
    env_name = _option("env_name", mlff_cfg, wsl=True) or DEFAULT_WSL_ENV
    micromamba = _option("micromamba", mlff_cfg, wsl=True)
    distro = _option("distro", mlff_cfg, wsl=True)
    wsl_root = _option("project_root", mlff_cfg, wsl=True)
    if not wsl_root:
        # El bloque WSL del runner DFT ya declara la raiz del repo; reusarla
        # evita tener que configurarla dos veces para la misma maquina.
        runner_wsl = ((config or {}).get("discovery", {}) or {}).get("wsl", {}) or {}
        wsl_root = _clean(runner_wsl.get("project_root")) or _windows_path_to_wsl(root)
    if not distro:
        runner_wsl = ((config or {}).get("discovery", {}) or {}).get("wsl", {}) or {}
        distro = _clean(runner_wsl.get("distro"))

    if backend == "auto":
        if sys.platform == "win32":
            backend = "wsl"
        elif _local_modules_present():
            backend = "local"
        else:
            backend = "wsl" if wsl_python else "local"

    if backend == "off":
        return MLFFRuntime(backend="off", env_name=env_name, timeout=timeout,
                           chunk_size=chunk, timeout_fases=timeout_fases)

    if backend == "wsl":
        if not wsl_python:
            wsl_python = f"/home/{_wsl_user_guess()}/perovowl-micromamba/envs/{env_name}/bin/python"
        return MLFFRuntime(
            backend="wsl",
            python=wsl_python,
            worker=f"{str(wsl_root).rstrip('/')}/{WORKER_REL}",
            distro=distro,
            project_root=str(wsl_root).rstrip("/"),
            env_name=env_name,
            micromamba=micromamba,
            timeout=timeout,
            chunk_size=chunk,
            timeout_fases=timeout_fases,
        )

    local_python = _option("python", mlff_cfg, wsl=False) or sys.executable
    worker = _option("worker", mlff_cfg, wsl=False) or str(root / WORKER_REL)
    return MLFFRuntime(
        backend="local",
        python=local_python,
        worker=worker,
        project_root=str(root),
        env_name=env_name,
        timeout=timeout,
        chunk_size=chunk,
        timeout_fases=timeout_fases,
    )


def _wsl_user_guess() -> str:
    """Usuario por defecto de la distro, solo para sugerir una ruta.

    Si falla, `probe()` dira que el interprete no existe y el wizard ofrecera
    crearlo: adivinar mal aqui no rompe nada.
    """
    try:
        proc = subprocess.run(
            ["wsl.exe", "--", "bash", "-lc", "echo $USER"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        user = (proc.stdout or "").strip().splitlines()
        if user and user[-1].strip():
            return user[-1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "root"
