"""Resolución de rutas: paquete, datos y configuración.

Hasta ahora todo el monitor calculaba una única raíz con
`Path(__file__).resolve().parents[N]`, asumiendo que se ejecuta desde un
checkout del repositorio. Eso deja de ser cierto en un ejecutable congelado:
`__file__` apunta al directorio temporal donde PyInstaller extrae el paquete.

Se separan por tanto tres raíces con ciclos de vida distintos:

* **bundle** — recursos de solo lectura que viajan con el programa (el SPA
  compilado, el ejemplo de configuración, las estructuras de referencia, los
  modelos del surrogate). Congelado es `sys._MEIPASS`; desde el código fuente,
  la raíz del repositorio.
* **data** — los datos del proyecto, que son del usuario y viven fuera del
  programa: `runs/`, `calculations/`, `reports/`, `imagenes/`, `data/` y los
  `scripts/` que lanza el control de batches.
* **config** — configuración y auditoría. Congelado va al directorio estándar
  del sistema; desde el repositorio se queda en `configs/`, como siempre.

Este módulo es **el único** de `monitor_api` autorizado a usar
`Path(__file__).parents[N]`; `tests/test_packaging.py` comprueba que no
reaparezca en ningún otro sitio.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = "dft-monitor"

# Pistas de que un directorio es la raíz de datos de un proyecto DFT.
MARCADORES_PROYECTO = ("runs", "local_runs", "calculations")

_data_root: Path | None = None


# ── Modo de ejecución ────────────────────────────────────────────────────────

def is_frozen() -> bool:
    """True si corre dentro de un ejecutable de PyInstaller."""
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def _repo_root() -> Path:
    """Raíz del repositorio a partir de la ubicación de este archivo."""
    return Path(__file__).resolve().parents[2]


# ── Raíz del paquete ─────────────────────────────────────────────────────────

def bundle_root() -> Path:
    """Recursos de solo lectura que acompañan al programa."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return _repo_root()


def bundle_file(*partes: str) -> Path:
    """Ruta a un recurso empaquetado, exista o no."""
    return bundle_root().joinpath(*partes)


def find_resource(*partes: str) -> Path:
    """Recurso que puede venir del paquete o de los datos del usuario.

    `structures/` y `models/` se empaquetan en el binario, pero cuando se
    ejecuta desde el repositorio son parte del árbol de datos. Se prefiere el
    paquete y se cae a los datos, de modo que un usuario puede sobreescribir un
    modelo dejando el suyo en su `data_root`.
    """
    empaquetado = bundle_root().joinpath(*partes)
    if empaquetado.exists():
        return empaquetado
    return data_root().joinpath(*partes)


# ── Raíz de datos ────────────────────────────────────────────────────────────

def set_data_root(path: str | Path, *, override: bool = True) -> Path:
    """Fija la raíz de datos.

    `override=True` corresponde a `--data-root` y gana siempre. `override=False`
    es para el valor del YAML, que solo aplica si no hay nada de más prioridad
    (el flag o la variable de entorno).
    """
    global _data_root
    if not override and (_data_root is not None or os.environ.get("DFT_DATA_ROOT")):
        return data_root()
    _data_root = Path(path).expanduser().resolve()
    return _data_root


def reset_data_root() -> None:
    """Olvida la raíz fijada (para tests)."""
    global _data_root
    _data_root = None


#: Raíz de datos por defecto del binario cuando no hay ninguna pista. Igual que
#: la que fija el paquete .deb en Linux, para que ambos coincidan.
CARPETA_DATOS_USUARIO = "PEROVOWL-data"


def data_root() -> Path:
    """Raíz de los datos del proyecto.

    Orden: `--data-root` → `DFT_DATA_ROOT` → el repositorio si se ejecuta desde
    el código fuente → un directorio con pinta de proyecto hacia arriba desde el
    actual → `~/PEROVOWL-data`.
    """
    if _data_root is not None:
        return _data_root

    entorno = os.environ.get("DFT_DATA_ROOT")
    if entorno:
        return Path(entorno).expanduser().resolve()

    # Desde el código fuente se conserva el comportamiento de siempre.
    if not is_frozen():
        return _repo_root()

    hallada = _buscar_raiz_proyecto()
    if hallada is not None:
        return hallada

    # Sin pistas: una carpeta por-usuario escribible, NUNCA el directorio de
    # trabajo. Al abrir el binario desde un acceso directo de Windows, el CWD es
    # `C:\Windows\System32`: ahí no se puede escribir y `config/generator.yaml`
    # no existe. `~/PEROVOWL-data` es lo mismo que usa el .deb.
    destino = (Path.home() / CARPETA_DATOS_USUARIO).resolve()
    try:
        destino.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return destino


def _buscar_raiz_proyecto() -> Path | None:
    """Directorio con pinta de proyecto hacia arriba desde el CWD, o None."""
    actual = Path.cwd().resolve()
    for candidato in (actual, *actual.parents):
        if any((candidato / m).is_dir() for m in MARCADORES_PROYECTO):
            return candidato
    return None


def resolve_data(rel: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz de datos.

    Las rutas absolutas se devuelven tal cual: la configuración admite ambas.
    """
    raw = str(rel)
    path = Path(rel).expanduser()
    if path.is_absolute() or (raw.startswith("/") and not raw.startswith("//")):
        return path
    return data_root() / path


# ── Configuración ────────────────────────────────────────────────────────────

def config_dir() -> Path:
    """Directorio de configuración y auditoría.

    Congelado usa la convención del sistema; desde el repositorio se queda en
    `configs/` para no cambiar el comportamiento actual.
    """
    forzado = os.environ.get("DFT_MONITOR_CONFIG_DIR")
    if forzado:
        return Path(forzado).expanduser().resolve()

    if not is_frozen():
        return _repo_root() / "configs"

    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")

    return Path(base).expanduser().resolve() / APP_NAME


def config_file() -> Path:
    return config_dir() / "monitor.yaml"


def example_config() -> Path:
    """El ejemplo versionado siempre viaja con el programa."""
    return bundle_file("configs", "monitor.example.yaml")


def audit_file() -> Path:
    return config_dir() / "monitor_audit.jsonl"


# ── Diagnóstico ──────────────────────────────────────────────────────────────

def describe() -> dict[str, str | bool]:
    """Resumen de las rutas efectivas, para `GET /api/health` y el arranque."""
    return {
        "frozen": is_frozen(),
        "bundle_root": str(bundle_root()),
        "data_root": str(data_root()),
        "config_dir": str(config_dir()),
    }


# ── Pipeline en fuente ───────────────────────────────────────────────────────

#: Sello con la version que materializo el pipeline. Al actualizar la app hay
#: que refrescar los fuentes: si no, un binario nuevo lanzaria runners viejos.
_SELLO_PIPELINE = ".pipeline-version"


def _pipeline_bundle() -> Path | None:
    """Los fuentes del pipeline dentro del binario, si viajan ahi."""
    if not is_frozen():
        return None
    candidato = bundle_root() / "pipeline"
    return candidato if candidato.is_dir() else None


def materializar_pipeline(*, version: str, forzar: bool = False) -> dict[str, object]:
    """Copia `scripts/` y `src/` del bundle a la raiz de datos.

    El runner de DFT no corre dentro de este proceso: es un Python externo —en
    Windows, el de WSL con GPAW— que hace `sys.path.insert(ROOT/"src")` y
    `from buho import ...`. Nadie puede importar desde dentro del archivo de
    PyInstaller, asi que los fuentes tienen que existir como ficheros. Sin esto
    `runner_launch_available` daba False en toda instalacion binaria: la app
    monitorizaba y cribaba, pero no podia lanzar un solo calculo.

    `version` se recibe en vez de importarse: este modulo tiene que poder
    importarse antes que cualquier otro del paquete, sin ciclos.

    Desde el repositorio no hace nada: los fuentes ya estan donde toca.
    """
    origen = _pipeline_bundle()
    if origen is None:
        return {"materializado": False, "motivo": "no congelado o sin pipeline en el bundle"}

    destino = data_root()
    sello = destino / _SELLO_PIPELINE
    actual = None
    if sello.is_file():
        try:
            actual = sello.read_text(encoding="utf-8").strip()
        except OSError:
            actual = None
    if actual == version and not forzar:
        return {"materializado": False, "motivo": "ya al dia", "version": actual}

    copiados = 0
    try:
        for sub in ("scripts", "src"):
            raiz_origen = origen / sub
            if not raiz_origen.is_dir():
                continue
            for fichero in raiz_origen.rglob("*"):
                if not fichero.is_file():
                    continue
                rel = fichero.relative_to(origen)
                final = destino / rel
                final.parent.mkdir(parents=True, exist_ok=True)
                # `copyfile` y no `copy2`: preservar la mtime del bundle haria
                # que un fichero recien escrito pareciera viejo.
                shutil.copyfile(fichero, final)
                copiados += 1
        sello.write_text(version, encoding="utf-8")
    except OSError as exc:
        # Que no se pueda escribir no debe impedir arrancar el monitor: sin
        # pipeline se puede seguir cribando y observando, y `runner_launch`
        # lo reportara como no disponible.
        log.warning("no se pudo materializar el pipeline en %s: %s", destino, exc)
        return {"materializado": False, "motivo": f"{type(exc).__name__}: {exc}"}

    log.info("pipeline materializado en %s (%d ficheros, v%s)", destino, copiados, version)
    return {"materializado": True, "ficheros": copiados, "destino": str(destino),
            "version": version}
