"""Diagnostico y reparacion del entorno: el motor del wizard.

PEROVOWL no corre en un solo interprete. Hay al menos tres runtimes y cada uno
puede faltar por su cuenta:

  * el del monitor (Windows `.venv-win` o el binario congelado): API, cascada
    Tier 0/1, GUI;
  * el de GPAW (WSL, env `gpaw246`): los calculos DFT;
  * el de MLFF (WSL, env propio): Tier 2 de la cascada, con torch/matgl.

Cuando falta uno, el sintoma llega tarde y disfrazado -- un `ModuleNotFoundError`
en mitad de una ronda, con el estado ya a medias. Este modulo existe para que la
comprobacion sea barata, explicita y anterior al fallo, y para que la
reparacion sea un paso ejecutable en vez de un parrafo de README.

No importa `click` a proposito: lo usan igual el CLI (`buho setup`) y la API
(`/api/setup/*`). Poner la logica en el CLI habria obligado a la API a
shellear su propio proceso para instalar algo.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from buho import mlff_runtime

#: Rueda de torch por defecto. La CPU basta: MEGNet/M3GNet sobre estas celdas
#: tardan ~0.5 s por candidato en un core, y la rueda CUDA pesa 3 veces mas.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

#: Rutas que empiezan por $HOME se dejan sin comillas para que bash las expanda.
#: Solo se acepta el prefijo literal y un juego de caracteres cerrado: cualquier
#: otra cosa (incluido lo que venga de la config) se cita como siempre.
_HOME_SEGURO = re.compile(r"^\$HOME(/[A-Za-z0-9._-]+)*$")


def _sh(ruta: str) -> str:
    """Cita una ruta para el shell, respetando un `$HOME` inicial."""
    return ruta if _HOME_SEGURO.match(ruta) else shlex.quote(ruta)

#: Version de Python del entorno MLFF. Fijada porque matgl publica ruedas por
#: version y 3.13+ todavia no siempre las tiene.
MLFF_PYTHON = "3.12"

#: Lo que necesita el worker MLFF, mas alla de torch.
MLFF_PACKAGES = ("matgl>=4.0", "pymatgen>=2024.1", "ase>=3.23", "pandas>=2.0",
                 "scikit-learn>=1.8", "pyyaml>=6.0")

#: Version de GPAW del entorno DFT. Fijada a la que se ha validado contra este
#: pipeline; "la ultima" ha roto el runner mas de una vez.
GPAW_VERSION = "24.6"

#: Python y numpy del entorno GPAW. numpy queda en 1.26 a proposito: GPAW 24.6
#: no compila contra la ABI de numpy 2, y es el motivo de que el entorno MLFF
#: viva aparte en vez de compartir este.
GPAW_PYTHON = "3.12"
GPAW_NUMPY = "1.26"

#: Nombre por defecto del entorno DFT en WSL.
GPAW_ENV = "gpaw246"

#: Distro que se propone cuando no hay ninguna instalada.
DISTRO_SUGERIDA = "Ubuntu"

GRUPOS_PIP = {
    "web": ("fastapi>=0.110", "uvicorn[standard]>=0.27", "httpx>=0.27",
            "psutil>=5.9", "itsdangerous>=2.1"),
    "desktop": ("pywebview>=5.0", "qtpy>=2.4", "PyQt6>=6.7", "PyQt6-WebEngine>=6.7"),
}


# ── Pasos ejecutables ─────────────────────────────────────────────────────────


@dataclass
class Step:
    """Un comando del plan de instalacion.

    `argv` ya viene resuelto para la maquina donde se va a ejecutar: si el
    destino es WSL, incluye el `wsl.exe -d ... -- bash -lc ...` por delante.
    """

    name: str
    argv: list[str]
    descripcion: str = ""
    #: Un fallo aqui no invalida el resto del plan (p. ej. un `clean` previo).
    opcional: bool = False
    timeout: int = 3600

    def shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "descripcion": self.descripcion,
            "comando": self.shell(),
            "opcional": self.opcional,
        }


@dataclass
class Plan:
    target: str
    steps: list[Step] = field(default_factory=list)
    notas: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "steps": [s.as_dict() for s in self.steps],
            "notas": self.notas,
        }


def execute(plan: Plan, *, on_output: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Ejecuta el plan en orden, transmitiendo la salida linea a linea.

    Se para en el primer paso obligatorio que falle: encadenar instalaciones
    sobre un entorno que no se pudo crear solo produce ruido.
    """
    emit = on_output or (lambda _linea: None)
    resultados: list[dict[str, Any]] = []

    for step in plan.steps:
        emit(f"$ {step.shell()}")
        try:
            proc = subprocess.Popen(
                step.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            resultados.append({"name": step.name, "ok": False, "error": str(exc)})
            emit(f"[error] {exc}")
            if not step.opcional:
                return {"status": "error", "steps": resultados,
                        "error": f"{step.name}: {exc}"}
            continue

        lineas: list[str] = []
        assert proc.stdout is not None
        for linea in proc.stdout:
            linea = linea.rstrip()
            lineas.append(linea)
            emit(linea)
        code = proc.wait(timeout=step.timeout)

        ok = code == 0
        resultados.append({
            "name": step.name,
            "ok": ok,
            "returncode": code,
            # La cola basta para diagnosticar y evita mandar 20 MB de pip a la GUI.
            "tail": lineas[-40:],
        })
        if not ok and not step.opcional:
            return {"status": "error", "steps": resultados,
                    "error": f"{step.name} salio con codigo {code}"}

    return {"status": "ok", "steps": resultados}


# ── Comprobacion ──────────────────────────────────────────────────────────────


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _wsl_disponible() -> bool:
    if sys.platform != "win32":
        return False
    from shutil import which

    return which("wsl.exe") is not None


def _run_wsl(distro: str | None, script: str, *, timeout: int = 60) -> subprocess.CompletedProcess | None:
    cmd = ["wsl.exe"]
    if distro:
        cmd.extend(["-d", distro])
    cmd.extend(["--", "bash", "-lc", script])
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


#: Donde suele vivir un Python con GPAW dentro de WSL. Se prueban en orden y
#: gana el primero que IMPORTE gpaw de verdad --- existir el fichero no basta,
#: un entorno a medio crear tiene el binario y no el paquete.
_CANDIDATOS_GPAW_WSL = (
    "$HOME/perovowl-micromamba/envs/*/bin/python",
    "$HOME/micromamba/envs/*/bin/python",
    "$HOME/miniconda3/envs/*/bin/python",
    "$HOME/anaconda3/envs/*/bin/python",
    "$HOME/mambaforge/envs/*/bin/python",
    "$HOME/miniforge3/envs/*/bin/python",
    "/opt/conda/envs/*/bin/python",
)


def sondear_gpaw_wsl(distro: str | None = None, *, timeout: int = 120
                     ) -> dict[str, Any]:
    """Mira que hay de GPAW en WSL y **cuenta lo que vio**.

    Por que existe
    --------------
    La comprobacion de capacidad decia "Define discovery.wsl.python en
    config/generator.yaml" en cuanto la configuracion no traia la ruta. Pero que
    no este configurado no significa que no este instalado: en una maquina donde
    GPAW ya vive en WSL, lo unico que faltaba era mirar.

    Y cuando de verdad no hay nada que encontrar, ese mismo mensaje es un
    consejo para un problema distinto: mandar a editar un YAML a quien lo que le
    falta son 2.5 GB de descarga. Por eso esto devuelve el sondeo entero
    --- si hay wsl.exe, que distros, que interpretes se probaron y con que
    resultado--- y no solo el hallazgo: sin eso, "no lo encontre" y "no mire" se
    ven igual desde fuera.

    Solo LEE. Nunca instala: eso se ofrece aparte, no se hace a escondidas.
    """
    sondeo: dict[str, Any] = {
        "wsl": _wsl_disponible(), "distro": distro, "distros": [],
        "candidatos": [], "probados": [], "hallado": None, "motivo": None,
    }
    if not sondeo["wsl"]:
        sondeo["motivo"] = "no hay wsl.exe en esta maquina"
        return sondeo

    # `wsl.exe` viene de serie en Windows 11 aunque no haya ninguna distro
    # instalada. Sin esta comprobacion, "WSL vacio" se confundia con "WSL con
    # Ubuntu pero sin GPAW", que son dos arreglos completamente distintos:
    # `wsl --install -d Ubuntu` frente a instalar el runtime desde Entorno.
    sondeo["distros"] = distros_wsl()
    if not sondeo["distros"]:
        sondeo["motivo"] = "hay wsl.exe pero ninguna distro instalada"
        return sondeo

    # OJO: nada de variables de shell aqui. Al pasar el script por
    # `wsl.exe -- bash -c`, `$algo` llega vacio --- incluso entre comillas
    # simples--- mientras que las del entorno como $HOME si sobreviven. Un bucle
    # `for p in ...; do ... "$p" ...; done` itera bien pero con la variable
    # vacia, y la deteccion devolvia None en una maquina donde GPAW SI estaba.
    # Por eso se listan los candidatos en una llamada y se prueban desde Python.
    listado = _run_wsl(distro, "ls -d " + " ".join(_CANDIDATOS_GPAW_WSL)
                       + " 2>/dev/null; command -v python3", timeout=timeout)
    if listado is None:
        sondeo["motivo"] = "wsl.exe no respondio (se agoto el tiempo o fallo al arrancar)"
        return sondeo
    candidatos = [c.strip() for c in (listado.stdout or "").splitlines() if c.strip()]
    sondeo["candidatos"] = candidatos
    if not candidatos:
        sondeo["motivo"] = "no hay ningun interprete de Python donde mirar"
        return sondeo

    for python in candidatos:
        proc = _run_wsl(
            distro,
            shlex.quote(python) + " -c "
            + shlex.quote("import gpaw,ase;print(gpaw.__version__,ase.__version__)"),
            timeout=timeout,
        )
        versiones = (proc.stdout or "").split() if proc is not None else []
        if proc is None or proc.returncode != 0 or not versiones:
            fallo = ((proc.stderr or "").strip().splitlines() or [""])[-1] if proc else "sin respuesta"
            sondeo["probados"].append({"python": python, "gpaw": None, "error": fallo[:200]})
            continue
        prefijo = python.rsplit("/bin/", 1)[0] if "/bin/" in python else None
        hallado: dict[str, Any] = {
            "python": python,
            "gpaw": versiones[0],
            "ase": versiones[1] if len(versiones) > 1 else None,
            "prefix": prefijo,
            "mpirun": f"{prefijo}/bin/mpiexec" if prefijo else None,
            "distro": distro,
        }
        if prefijo:
            # `gpaw-data` de conda-forge deja los setups en site-packages, no en
            # share/gpaw. Comprobado sobre un entorno real.
            hallado["setup_path"] = (
                f"{prefijo}/lib/python{GPAW_PYTHON}/site-packages/gpaw_data/setups")
        sondeo["probados"].append({"python": python, "gpaw": versiones[0]})
        sondeo["hallado"] = hallado
        return sondeo

    sondeo["motivo"] = (
        f"se probaron {len(candidatos)} interpretes y ninguno importa GPAW")
    return sondeo


def detectar_gpaw_wsl(distro: str | None = None, *, timeout: int = 120
                      ) -> dict[str, Any] | None:
    """El GPAW ya instalado en WSL, o None. `sondear_gpaw_wsl` dice por que."""
    return sondear_gpaw_wsl(distro, timeout=timeout)["hallado"]


def _capacidad(nombre: str, titulo: str, ok: bool, *, detalle: Any = None,
               error: str | None = None, remediacion: str = "",
               comando: str | None = None, requerido: bool = True) -> dict[str, Any]:
    return {
        "id": nombre,
        "titulo": titulo,
        "ok": bool(ok),
        "requerido": requerido,
        "detalle": detalle or {},
        "error": error,
        "remediacion": remediacion,
        "comando": comando,
    }


def _check_core() -> dict[str, Any]:
    modulos = ("ase", "numpy", "pandas", "yaml", "click", "sklearn")
    faltan = [m for m in modulos if not _importable(m)]
    return _capacidad(
        "core", "Núcleo del pipeline", not faltan,
        detalle={
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "modulos": {m: _importable(m) for m in modulos},
        },
        error=None if not faltan else f"Faltan módulos: {', '.join(faltan)}",
        remediacion="" if not faltan else "Instala el paquete con 'pip install -e .'",
        comando=None if not faltan else "buho setup install core",
    )


def _check_web() -> dict[str, Any]:
    modulos = ("fastapi", "uvicorn", "httpx", "psutil", "itsdangerous")
    faltan = [m for m in modulos if not _importable(m)]
    return _capacidad(
        "web", "API del monitor", not faltan,
        detalle={"modulos": {m: _importable(m) for m in modulos}},
        error=None if not faltan else f"Faltan módulos: {', '.join(faltan)}",
        remediacion="" if not faltan else "El monitor no puede servir la API sin esto.",
        comando=None if not faltan else "buho setup install web",
    )


def _check_paw(project_root: Path | None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Setups PAW. En Windows hay que mirarlos donde corre GPAW, no aquí.

    Buscarlos en el sistema de ficheros de Windows daba un rojo permanente en
    una máquina cuyo DFT funciona perfectamente: los datasets viven en el env
    de WSL, que es el único sitio donde GPAW los va a abrir.
    """
    discovery = (config or {}).get("discovery", {}) or {}
    wsl_cfg = discovery.get("wsl", {}) or {}
    setup_path = wsl_cfg.get("setup_path")

    if sys.platform == "win32" and setup_path and _wsl_disponible():
        distro = wsl_cfg.get("distro")
        marcador = shlex.quote(f"{str(setup_path).rstrip('/')}/Cs.PBE.gz")
        proc = _run_wsl(distro, f"test -d {shlex.quote(str(setup_path))} && ls {marcador}")
        ok = proc is not None and proc.returncode == 0
        return _capacidad(
            "paw", "Datasets PAW de GPAW (WSL)", ok,
            detalle={"ruta": setup_path, "backend": "wsl", "distro": distro},
            error=None if ok else f"No se encontraron setups PAW en {setup_path}.",
            remediacion="" if ok else "gpaw install-data <ruta> --register dentro de WSL.",
            requerido=False,
        )

    try:
        from buho import gpaw_setup

        encontrado = gpaw_setup.find(project_root)
    except Exception as exc:  # noqa: BLE001
        return _capacidad("paw", "Datasets PAW de GPAW", False,
                          error=f"{type(exc).__name__}: {exc}",
                          remediacion="Revisa GPAW_SETUP_PATH.", requerido=False)
    return _capacidad(
        "paw", "Datasets PAW de GPAW", bool(encontrado),
        detalle={"ruta": encontrado},
        error=None if encontrado else "No se encontró un directorio de setups PAW.",
        remediacion="" if encontrado else "gpaw install-data ~/.gpaw/datasets --register",
        requerido=False,
    )


def _check_dft(config: dict[str, Any] | None) -> dict[str, Any]:
    """Runtime GPAW. En Windows vive en WSL; aquí solo se comprueba, no se toca."""
    discovery = (config or {}).get("discovery", {}) or {}
    wsl_cfg = discovery.get("wsl", {}) or {}
    distro = wsl_cfg.get("distro")
    python = wsl_cfg.get("python")

    if sys.platform != "win32":
        ok = _importable("gpaw")
        return _capacidad("dft", "Runtime DFT (GPAW)", ok,
                          detalle={"backend": "local", "gpaw": _importable("gpaw")},
                          error=None if ok else "GPAW no es importable en este intérprete.",
                          remediacion="" if ok else "pip install gpaw")

    if not _wsl_disponible():
        return _capacidad("dft", "Runtime DFT (GPAW en WSL)", False,
                          detalle={"backend": "wsl", "distro": distro},
                          error="No se encontró wsl.exe.",
                          remediacion="Instala WSL: 'wsl --install'.")
    if not python:
        return _capacidad("dft", "Runtime DFT (GPAW en WSL)", False,
                          detalle={"backend": "wsl", "distro": distro},
                          error="No hay intérprete GPAW configurado.",
                          remediacion="Define discovery.wsl.python en config/generator.yaml.")

    proc = _run_wsl(distro, f"{shlex.quote(str(python))} -c "
                            "'import gpaw,ase;print(gpaw.__version__,ase.__version__)'")
    ok = proc is not None and proc.returncode == 0
    # `versiones` es un dict en TODAS las capacidades. Devolverlo aquí como el
    # string crudo de stdout hacía que la GUI, que itera el mapa, recorriera los
    # caracteres del texto y pintara un chip por cada uno.
    salida = (proc.stdout or "").strip().split() if proc else []
    versiones = dict(zip(("gpaw", "ase"), salida))
    return _capacidad(
        "dft", "Runtime DFT (GPAW en WSL)", ok,
        detalle={"backend": "wsl", "distro": distro, "python": python,
                 "versiones": versiones},
        error=None if ok else ((proc.stderr or "").strip()[:300] if proc else "WSL no respondió."),
        remediacion="" if ok else "Revisa el entorno gpaw246 en WSL.",
    )


def _check_mlff(config: dict[str, Any] | None, project_root: Path | None) -> dict[str, Any]:
    try:
        runtime = mlff_runtime.resolve(config, project_root=project_root)
    except mlff_runtime.MLFFUnavailableError as exc:
        return _capacidad("mlff", "Cribado MLFF/GNN (Tier 2)", False,
                          error=str(exc), remediacion=exc.remediation,
                          comando="buho setup install mlff", requerido=False)

    sonda = runtime.probe()
    ok = bool(sonda.get("available"))
    return _capacidad(
        "mlff", "Cribado MLFF/GNN (Tier 2)", ok,
        detalle={
            "backend": runtime.backend,
            "python": runtime.python,
            "distro": runtime.distro,
            "env_name": runtime.env_name,
            "versiones": sonda.get("versions", {}),
        },
        error=sonda.get("error"),
        remediacion=sonda.get("remediation") or "",
        comando=None if ok else "buho setup install mlff",
        # Sin Tier 2 la cascada sigue cribando con Tier 0/1: es una degradación,
        # no una parada. Marcarlo como requerido pintaría de rojo un monitor
        # que funciona.
        requerido=False,
    )


def check(config: dict[str, Any] | None = None, *,
          project_root: Path | str | None = None,
          incluir_mlff: bool = True) -> dict[str, Any]:
    """Matriz de capacidades de esta máquina.

    Las comprobaciones que arrancan WSL (DFT, PAW, MLFF) corren **en paralelo**.
    Cada invocación de `wsl.exe` cuesta unos segundos y son independientes
    entre sí; en serie sumaban ~15 s, que es tiempo de spinner en la pantalla de
    Entorno. Son I/O puro esperando a un subproceso, así que hilos bastan.

    `incluir_mlff=False` se salta la sonda MLFF (la más cara de las tres).
    """
    root = Path(project_root) if project_root else None

    # Las dos baratas son sólo `find_spec` en este intérprete: no compensa
    # mandarlas al pool.
    capacidades = [_check_core(), _check_web()]

    tareas: list[Callable[[], dict[str, Any]]] = [
        lambda: _check_dft(config),
        lambda: _check_paw(root, config),
    ]
    if incluir_mlff:
        tareas.append(lambda: _check_mlff(config, root))

    with ThreadPoolExecutor(max_workers=len(tareas)) as pool:
        # `map` conserva el orden de envío, que es el que espera la GUI.
        capacidades.extend(pool.map(lambda f: f(), tareas))

    requeridas_ok = all(c["ok"] for c in capacidades if c["requerido"])
    todas_ok = all(c["ok"] for c in capacidades)
    return {
        "status": "ok" if todas_ok else ("degradado" if requeridas_ok else "error"),
        "ok": requeridas_ok,
        "plataforma": sys.platform,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)),
        "capacidades": capacidades,
    }


# ── Planes de instalacion ─────────────────────────────────────────────────────


def _pip_local(paquetes: Iterable[str], *, extra: list[str] | None = None) -> list[str]:
    return [sys.executable, "-m", "pip", "install", "--upgrade", *(extra or []), *paquetes]


def _mlff_paths(config: dict[str, Any] | None, env_name: str) -> dict[str, str]:
    """Rutas del entorno MLFF en WSL, derivadas de donde ya vive micromamba."""
    discovery = (config or {}).get("discovery", {}) or {}
    mlff_cfg = discovery.get("mlff", {}) or {}
    wsl_cfg = mlff_cfg.get("wsl", {}) or {}

    micromamba = wsl_cfg.get("micromamba") or os.environ.get("BUHO_WSL_MICROMAMBA")
    if not micromamba:
        # El runner de DFT ya apunta a un python dentro de <root>/envs/<env>/bin,
        # asi que la raiz de micromamba se deduce de ahi sin preguntar nada.
        gpaw_python = (discovery.get("wsl", {}) or {}).get("python", "")
        if "/envs/" in str(gpaw_python):
            raiz = str(gpaw_python).split("/envs/")[0]
            micromamba = f"{raiz}/bin/micromamba"
        else:
            # Sin nada de donde deducir (maquina limpia): ruta absoluta bajo
            # $HOME. Un "micromamba" pelado dejaria el binario en el directorio
            # actual y el resto del plan no lo encontraria.
            micromamba = "$HOME/perovowl-micromamba/bin/micromamba"

    raiz = micromamba.rsplit("/bin/", 1)[0] if "/bin/" in micromamba else "$HOME/perovowl-micromamba"
    return {
        "micromamba": micromamba,
        "root_prefix": raiz,
        "env_prefix": f"{raiz}/envs/{env_name}",
        "python": f"{raiz}/envs/{env_name}/bin/python",
        "pip": f"{raiz}/envs/{env_name}/bin/pip",
    }


def plan_mlff(config: dict[str, Any] | None = None, *,
              env_name: str | None = None,
              distro: str | None = None,
              cuda: bool = False,
              recrear: bool = False) -> Plan:
    """Crea (o repara) el entorno MLFF en WSL.

    Entorno **separado** de `gpaw246` a propósito: GPAW está fijado a numpy 1.26
    y matgl arrastra numpy>=2. Compartirlos rompería el DFT que ya funciona,
    que es justo lo que este proyecto no se puede permitir perder.
    """
    discovery = (config or {}).get("discovery", {}) or {}
    mlff_cfg = discovery.get("mlff", {}) or {}
    env = env_name or (mlff_cfg.get("wsl", {}) or {}).get("env_name") or mlff_runtime.DEFAULT_WSL_ENV
    distro = distro or (mlff_cfg.get("wsl", {}) or {}).get("distro") or \
        (discovery.get("wsl", {}) or {}).get("distro")

    rutas = _mlff_paths(config, env)
    mm, root_prefix = rutas["micromamba"], rutas["root_prefix"]
    env_python, env_pip = rutas["python"], rutas["pip"]

    def wsl_step(name: str, script: str, *, descripcion: str = "",
                 opcional: bool = False, timeout: int = 3600) -> Step:
        cmd = ["wsl.exe"]
        if distro:
            cmd.extend(["-d", str(distro)])
        cmd.extend(["--", "bash", "-lc", script])
        return Step(name=name, argv=cmd, descripcion=descripcion,
                    opcional=opcional, timeout=timeout)

    steps: list[Step] = []
    # micromamba puede no existir: aquí ya estaba porque el entorno de GPAW se
    # creó con él, pero en una máquina limpia el plan fallaba en el primer paso
    # con un "command not found" que no decía qué hacer. Se descarga si falta;
    # si está, el `test -x` corta y no se toca nada.
    steps.append(wsl_step(
        "asegurar-micromamba",
        f"test -x {_sh(mm)} || {{ "
        f"mkdir -p {_sh(os.path.dirname(mm))} && "
        f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        f"| tar -xvj -C /tmp bin/micromamba && "
        f"mv /tmp/bin/micromamba {_sh(mm)} && "
        f"chmod +x {_sh(mm)}; }}",
        descripcion="Comprueba micromamba y lo descarga solo si falta.",
        timeout=900,
    ))
    if recrear:
        steps.append(wsl_step(
            "limpiar",
            f"{_sh(mm)} env remove -y -r {_sh(root_prefix)} -n {shlex.quote(env)} || true",
            descripcion=f"Elimina el entorno {env} si existía.",
            opcional=True, timeout=600,
        ))

    steps.append(wsl_step(
        "crear-entorno",
        f"{_sh(mm)} create -y -r {_sh(root_prefix)} -n {shlex.quote(env)} "
        f"python={MLFF_PYTHON} pip",
        descripcion=f"Crea el entorno micromamba {env} con Python {MLFF_PYTHON}.",
        timeout=1800,
    ))

    indice = [] if cuda else ["--index-url", TORCH_CPU_INDEX]
    steps.append(wsl_step(
        "instalar-torch",
        " ".join(shlex.quote(p) for p in [env_pip, "install", "--upgrade", *indice, "torch"]),
        descripcion="Instala torch" + (" (CUDA)" if cuda else " (rueda CPU, ~200 MB)") + ".",
        timeout=3600,
    ))
    steps.append(wsl_step(
        "instalar-mlff",
        " ".join(shlex.quote(p) for p in [env_pip, "install", "--upgrade", *MLFF_PACKAGES]),
        descripcion="Instala matgl, pymatgen y el resto de lo que usa el worker.",
        timeout=3600,
    ))

    proyecto = (mlff_cfg.get("wsl", {}) or {}).get("project_root") or \
        (discovery.get("wsl", {}) or {}).get("project_root")
    if proyecto:
        worker = f"{str(proyecto).rstrip('/')}/{mlff_runtime.WORKER_REL}"
        steps.append(wsl_step(
            "verificar",
            f"{_sh(env_python)} {shlex.quote(worker)} --preflight-only",
            descripcion="Comprueba que el worker importa torch/matgl/pymatgen.",
            timeout=600,
        ))

    plan = Plan(target="mlff", steps=steps)
    plan.notas.append(
        "Entorno separado de gpaw246: GPAW usa numpy 1.26 y matgl exige numpy>=2."
    )
    plan.notas.append(f"Al terminar, apunta discovery.mlff.wsl.python a {env_python}.")
    if cuda:
        plan.notas.append("Rueda CUDA: ocupa ~3 GB más y solo compensa con GPU NVIDIA.")
    return plan


def distros_wsl() -> list[str]:
    """Distros instaladas. Lista vacia si WSL no esta o no hay ninguna."""
    if not _wsl_disponible():
        return []
    try:
        proc = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    # wsl.exe -l escribe UTF-16LE con NULs intercalados.
    # wsl.exe -l escribe UTF-16LE; si no lo es, viene en UTF-8.
    try:
        salida = proc.stdout.decode("utf-16-le")
    except UnicodeDecodeError:
        salida = proc.stdout.decode("utf-8", errors="replace")
    salida = salida.replace(chr(0), "")
    return [linea.strip() for linea in salida.splitlines() if linea.strip()]


def _rutas_gpaw(config: dict[str, Any] | None, env_name: str) -> dict[str, str]:
    """Rutas del entorno GPAW en WSL, reutilizando micromamba si ya existe."""
    discovery = (config or {}).get("discovery", {}) or {}
    wsl_cfg = discovery.get("wsl", {}) or {}

    micromamba = wsl_cfg.get("micromamba")
    if not micromamba:
        actual = str(wsl_cfg.get("python", ""))
        if "/envs/" in actual:
            micromamba = f"{actual.split('/envs/')[0]}/bin/micromamba"
        else:
            micromamba = "$HOME/perovowl-micromamba/bin/micromamba"
    raiz = micromamba.rsplit("/bin/", 1)[0] if "/bin/" in micromamba else "$HOME/perovowl-micromamba"
    prefijo = f"{raiz}/envs/{env_name}"
    return {
        "micromamba": micromamba,
        "root_prefix": raiz,
        "prefix": prefijo,
        "python": f"{prefijo}/bin/python",
        "mpirun": f"{prefijo}/bin/mpiexec",
        # Los datasets PAW llegan como paquete `gpaw-data` de conda-forge, que
        # los deja en site-packages, no en share/gpaw. Comprobado sobre un
        # entorno real: `share/gpaw` no existe. Apuntar ahi dejaba la config
        # senalando a un directorio vacio.
        "setups": f"{prefijo}/lib/python{GPAW_PYTHON}/site-packages/gpaw_data/setups",
    }


def plan_dft(config: dict[str, Any] | None = None, *,
             distro: str | None = None,
             env_name: str | None = None,
             recrear: bool = False) -> Plan:
    """Crea el entorno GPAW en WSL y descarga los datasets PAW.

    Instalar WSL en si NO se intenta: `wsl --install` exige privilegios de
    administrador y un reinicio. Una app cientifica que pide elevacion es
    intrusiva, y el fallo sin ella seria confuso. Se detecta y se da el comando
    exacto; a partir de que exista una distro, el resto si es automatico.
    """
    discovery = (config or {}).get("discovery", {}) or {}
    wsl_cfg = discovery.get("wsl", {}) or {}
    env = env_name or wsl_cfg.get("env_name") or GPAW_ENV

    plan = Plan(target="dft")

    if sys.platform != "win32":
        plan.notas.append(
            "Fuera de Windows, GPAW se instala en el sistema o en un entorno "
            "conda propio; este plan solo cubre el camino por WSL."
        )
        return plan

    if not _wsl_disponible():
        plan.notas.append(
            "No se encontro wsl.exe. Instalar WSL necesita permisos de "
            "administrador y reiniciar, asi que no se puede hacer desde aqui."
        )
        plan.notas.append(
            "Abre PowerShell como administrador y ejecuta:  wsl --install"
        )
        plan.notas.append("Reinicia y vuelve a esta pantalla.")
        return plan

    instaladas = distros_wsl()
    if not instaladas:
        plan.notas.append(
            "WSL esta pero no hay ninguna distribucion instalada."
        )
        plan.notas.append(
            f"En PowerShell:  wsl --install -d {DISTRO_SUGERIDA}"
        )
        plan.notas.append(
            "La primera vez pide crear usuario y contrasena de forma "
            "interactiva, por eso no se lanza desde la app."
        )
        return plan

    # Se reutiliza la distro que el usuario ya tenga antes de proponer otra.
    distro = distro or wsl_cfg.get("distro") or instaladas[0]
    rutas = _rutas_gpaw(config, env)
    mm, root_prefix = rutas["micromamba"], rutas["root_prefix"]

    def wsl_step(name: str, script: str, *, descripcion: str = "",
                 opcional: bool = False, timeout: int = 3600) -> Step:
        cmd = ["wsl.exe", "-d", str(distro), "--", "bash", "-lc", script]
        return Step(name=name, argv=cmd, descripcion=descripcion,
                    opcional=opcional, timeout=timeout)

    steps: list[Step] = []
    steps.append(wsl_step(
        "asegurar-micromamba",
        f"test -x {_sh(mm)} || {{ "
        f"mkdir -p {_sh(os.path.dirname(mm))} && "
        f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        f"| tar -xvj -C /tmp bin/micromamba && "
        f"mv /tmp/bin/micromamba {_sh(mm)} && chmod +x {_sh(mm)}; }}",
        descripcion="Comprueba micromamba y lo descarga solo si falta.",
        timeout=900,
    ))
    if recrear:
        steps.append(wsl_step(
            "limpiar",
            f"{_sh(mm)} env remove -y -r {_sh(root_prefix)} -n {shlex.quote(env)} || true",
            descripcion=f"Elimina el entorno {env} si existia.",
            opcional=True, timeout=900,
        ))
    steps.append(wsl_step(
        "crear-entorno",
        f"{_sh(mm)} create -y -r {_sh(root_prefix)} -n {shlex.quote(env)} "
        f"-c conda-forge python={GPAW_PYTHON} numpy={GPAW_NUMPY} "
        f"gpaw={GPAW_VERSION} ase openmpi",
        descripcion=(f"Crea {env} con GPAW {GPAW_VERSION}, numpy {GPAW_NUMPY} "
                     "y OpenMPI (~2 GB)."),
        timeout=5400,
    ))
    # `gpaw-data` entra como dependencia de conda-forge y deja los setups en
    # site-packages, asi que normalmente no hay nada que descargar. Se
    # comprueba con un elemento concreto —el mismo que usa la comprobacion de
    # capacidad— y solo si falta se recurre a `gpaw install-data`, que si baja
    # ~500 MB. Descargarlos siempre era gastar red y tiempo por costumbre.
    setups = rutas["setups"]
    steps.append(wsl_step(
        "datasets-paw",
        f"test -f {_sh(setups + '/Cs.PBE.gz')} "
        f"|| {_sh(root_prefix + '/envs/' + env + '/bin/gpaw')} install-data "
        f"--register {_sh(setups)}",
        descripcion=("Comprueba los datasets PAW y solo los descarga si "
                     "gpaw-data no los trajo (~500 MB en ese caso)."),
        timeout=3600,
    ))
    steps.append(wsl_step(
        "verificar",
        f"{_sh(rutas['python'])} -c "
        + shlex.quote("import gpaw, ase; print('gpaw', gpaw.__version__, 'ase', ase.__version__)"),
        descripcion="Comprueba que GPAW y ASE importan en el entorno nuevo.",
        timeout=600,
    ))

    plan.steps = steps
    plan.notas.append(f"Distro: {distro}. Entorno: {env}.")
    plan.notas.append(
        "Descarga unos 2.5 GB entre el entorno y los datasets PAW."
    )
    plan.notas.append(
        "Al terminar se escribe discovery.wsl en la configuracion del usuario, "
        "con project_root apuntando a la raiz de datos vista desde WSL."
    )
    return plan


def plan_pip(target: str) -> Plan:
    """Instala un grupo de extras en el intérprete actual."""
    paquetes = GRUPOS_PIP.get(target)
    if paquetes is None:
        raise ValueError(f"Grupo desconocido: {target}")
    if getattr(sys, "frozen", False):
        plan = Plan(target=target)
        plan.notas.append(
            "Este monitor es un binario congelado: no tiene site-packages donde "
            "instalar. Usa el entorno de desarrollo o el instalador del sistema."
        )
        return plan
    return Plan(
        target=target,
        steps=[Step(name=f"instalar-{target}", argv=_pip_local(paquetes),
                    descripcion=f"Instala los extras '{target}'.")],
    )


def plan(target: str, *, config: dict[str, Any] | None = None, **opciones: Any) -> Plan:
    if target == "mlff":
        return plan_mlff(config, **opciones)
    if target == "dft":
        return plan_dft(config, **opciones)
    if target in GRUPOS_PIP:
        return plan_pip(target)
    raise ValueError(f"Objetivo desconocido: {target}. Válidos: dft, mlff, "
                     + ", ".join(sorted(GRUPOS_PIP)))
