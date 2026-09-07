"""Requisitos para poder empaquetar el monitor como binario.

Congelado con PyInstaller, `__file__` apunta al directorio temporal de
extracción: cualquier ruta derivada de él deja de existir. Estos tests fijan las
reglas que lo hacen posible y evitan que se rompan por descuido.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAQUETE = ROOT / "src" / "monitor_api"
sys.path.insert(0, str(ROOT / "src"))

from monitor_api import paths  # noqa: E402

# `paths.py` es el único autorizado: es donde vive la definición.
EXENTOS = {"paths.py"}
ANCLAJE = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[")


@pytest.fixture(autouse=True)
def raiz_limpia(monkeypatch):
    monkeypatch.delenv("DFT_DATA_ROOT", raising=False)
    monkeypatch.delenv("DFT_MONITOR_CONFIG_DIR", raising=False)
    paths.reset_data_root()
    yield
    paths.reset_data_root()


def _congelar(monkeypatch, meipass: Path) -> None:
    """Simula la ejecución dentro de un ejecutable de PyInstaller."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)


# ── Desanclaje del repositorio ───────────────────────────────────────────────

def test_ningun_modulo_calcula_la_raiz_desde_su_ubicacion():
    """Regresión: eran catorce sitios y todos rompían al congelar."""
    culpables = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in sorted(PAQUETE.rglob("*.py"))
        if p.name not in EXENTOS
        for i, linea in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if ANCLAJE.search(linea)
    ]
    assert not culpables, (
        "Usa monitor_api.paths en vez de derivar la raíz de __file__:\n  "
        + "\n  ".join(culpables)
    )


#: `buho/` y `ml_surrogate/` sí anclan al repositorio vía `__file__`, y en la
#: mayoría de sitios es legítimo. Esta lista es un trinquete: fija los que hay
#: hoy, con su justificación, para que uno nuevo tenga que argumentarse.
#:
#: Congelado, `__file__` cuelga del directorio de extracción, así que
#: `parents[N]` apunta fuera del bundle. Sirve como *default* cuando el llamador
#: siempre pasa la raíz, o en código que solo corre por CLI desde el repo. NO
#: sirve para localizar recursos empaquetados: ahí hay que mirar `sys._MEIPASS`.
#: Regresión real: `bandgap_scissor` lo hacía y la corrección SOC quedó muerta
#: en todos los binarios publicados, en silencio.
ANCLAJE_PERMITIDO = {
    # Solo CLI, desde el repositorio: nunca se ejecutan dentro del binario.
    "buho/generator/__main__.py",
    "buho/dft_jobs/__main__.py",
    "buho/phase2_force/__init__.py",
    "buho/active_learning/batch_loop.py",
    "ml_surrogate/train.py",
    "ml_surrogate/predict.py",
    "ml_surrogate/inference.py",
    "ml_surrogate/structure_builder.py",
    # Default de una raíz que el llamador del monitor siempre pasa explícita.
    "buho/discovery/engine.py",
    "buho/mlff_runtime.py",
    "buho/generator/heuristic_generator.py",
    "ml_surrogate/config.py",
    # Consultan `sys._MEIPASS` primero; esto es el fallback desde fuentes.
    "buho/bandgap_scissor.py",
    "buho/eg_scale.py",
}


def test_el_anclaje_al_repo_fuera_de_monitor_api_no_crece():
    """Trinquete sobre `buho/` y `ml_surrogate/`.

    El test hermano cubre `monitor_api`, donde el anclaje está prohibido del
    todo. Aquí se tolera lo que ya existe y se bloquea lo nuevo.
    """
    src = ROOT / "src"
    encontrados = {
        p.relative_to(src).as_posix()
        for paquete in ("buho", "ml_surrogate")
        for p in sorted((src / paquete).rglob("*.py"))
        for linea in p.read_text(encoding="utf-8").splitlines()
        if ANCLAJE.search(linea)
    }

    nuevos = encontrados - ANCLAJE_PERMITIDO
    assert not nuevos, (
        "Anclaje nuevo a __file__. Congelado no resuelve dentro del bundle;\n"
        "usa sys._MEIPASS o recibe la raíz del llamador:\n  "
        + "\n  ".join(sorted(nuevos))
    )

    obsoletos = ANCLAJE_PERMITIDO - encontrados
    assert not obsoletos, (
        "Ya no anclan; quítalos de ANCLAJE_PERMITIDO:\n  " + "\n  ".join(sorted(obsoletos))
    )


def test_el_scissor_soc_se_resuelve_dentro_del_bundle(tmp_path, monkeypatch):
    """Regresión: la tabla no cargaba en el binario y no se corregía nada.

    `parents[2]` desde `_MEIPASS/buho/bandgap_scissor.py` cae fuera del bundle.
    """
    from buho import bandgap_scissor as bs

    mei = tmp_path / "engine" / "_internal"
    (mei / "buho").mkdir(parents=True)
    (mei / "config").mkdir()
    (mei / "config" / "soc_scissor.json").write_text(
        json.dumps({"chi_soc_eV": {"Pb": -0.63}}), encoding="utf-8"
    )

    _congelar(monkeypatch, mei)
    monkeypatch.setattr(bs, "__file__", str(mei / "buho" / "bandgap_scissor.py"))
    bs._cache.clear()
    try:
        assert bs.cargar_tabla() == {"Pb": -0.63}
        assert bs.corregir(1.20, {"Pb": 1.0}) == pytest.approx(0.57)
    finally:
        bs._cache.clear()


def test_el_spec_empaqueta_la_tabla_del_scissor():
    """Sin ella, `bandgap_scissor` devuelve tabla vacía y no corrige nada."""
    spec = (ROOT / "packaging" / "dft-monitor-engine.spec").read_text(encoding="utf-8")
    assert "soc_scissor.json" in spec


def test_paths_no_importa_nada_del_paquete():
    """Debe poder importarse antes que cualquier otra cosa, sin ciclos."""
    fuente = (PAQUETE / "paths.py").read_text(encoding="utf-8")
    assert "from ." not in fuente
    assert "from monitor_api" not in fuente


# ── Raíz de datos ────────────────────────────────────────────────────────────

def test_data_root_desde_el_codigo_fuente_es_el_repositorio():
    """El comportamiento de siempre no debe cambiar al ejecutar desde el repo."""
    assert paths.data_root() == ROOT


def test_set_data_root_tiene_prioridad(tmp_path):
    paths.set_data_root(tmp_path)
    assert paths.data_root() == tmp_path.resolve()


def test_la_variable_de_entorno_gana_al_yaml(tmp_path, monkeypatch):
    """--data-root > DFT_DATA_ROOT > monitor.data_root del YAML."""
    entorno = tmp_path / "entorno"
    yaml_dir = tmp_path / "yaml"
    entorno.mkdir()
    yaml_dir.mkdir()

    monkeypatch.setenv("DFT_DATA_ROOT", str(entorno))
    # override=False es como se aplica el valor del YAML.
    paths.set_data_root(yaml_dir, override=False)
    assert paths.data_root() == entorno.resolve()

    # El flag sí manda sobre ambos.
    flag = tmp_path / "flag"
    flag.mkdir()
    paths.set_data_root(flag)
    assert paths.data_root() == flag.resolve()


def test_el_yaml_aplica_si_no_hay_nada_mas(tmp_path):
    paths.set_data_root(tmp_path, override=False)
    assert paths.data_root() == tmp_path.resolve()


def test_congelado_busca_una_raiz_con_pinta_de_proyecto(tmp_path, monkeypatch):
    proyecto = tmp_path / "proyecto"
    (proyecto / "local_runs").mkdir(parents=True)
    hondo = proyecto / "a" / "b"
    hondo.mkdir(parents=True)

    _congelar(monkeypatch, tmp_path / "bundle")
    monkeypatch.chdir(hondo)

    assert paths.data_root() == proyecto.resolve()


def test_congelado_sin_pistas_usa_una_carpeta_del_usuario(tmp_path, monkeypatch):
    """Nunca el CWD: al abrir desde un acceso directo de Windows es System32.

    Regresión real: v0.4.0 daba `No se encuentra
    C:\\Windows\\System32\\config\\generator.yaml` en el cribado.
    """
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    casa = tmp_path / "home"
    casa.mkdir()
    _congelar(monkeypatch, tmp_path / "bundle")
    monkeypatch.chdir(vacio)
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: casa))

    raiz = paths.data_root()
    assert raiz == (casa / paths.CARPETA_DATOS_USUARIO).resolve()
    assert raiz.is_dir()          # se crea
    assert raiz != vacio.resolve()


def test_resolve_data_respeta_las_rutas_absolutas(tmp_path):
    paths.set_data_root(tmp_path)
    assert paths.resolve_data("runs/x") == tmp_path.resolve() / "runs" / "x"
    assert paths.resolve_data("/opt/datos") == Path("/opt/datos")


# ── Raíz del paquete ─────────────────────────────────────────────────────────

def test_bundle_root_usa_meipass_al_congelar(tmp_path, monkeypatch):
    meipass = tmp_path / "_MEI123"
    meipass.mkdir()
    _congelar(monkeypatch, meipass)

    assert paths.is_frozen() is True
    assert paths.bundle_root() == meipass


def test_bundle_root_desde_el_codigo_fuente_es_el_repositorio():
    assert paths.is_frozen() is False
    assert paths.bundle_root() == ROOT


def test_find_resource_prefiere_lo_empaquetado(tmp_path, monkeypatch):
    meipass = tmp_path / "bundle"
    datos = tmp_path / "datos"
    (meipass / "models").mkdir(parents=True)
    (datos / "models").mkdir(parents=True)

    _congelar(monkeypatch, meipass)
    paths.set_data_root(datos)

    assert paths.find_resource("models") == meipass / "models"


def test_find_resource_cae_a_los_datos_si_no_esta_empaquetado(tmp_path, monkeypatch):
    meipass = tmp_path / "bundle"
    datos = tmp_path / "datos"
    meipass.mkdir()
    (datos / "models").mkdir(parents=True)

    _congelar(monkeypatch, meipass)
    paths.set_data_root(datos)

    assert paths.find_resource("models") == datos / "models"


# ── Configuración ────────────────────────────────────────────────────────────

def test_config_dir_desde_el_codigo_fuente_es_configs():
    assert paths.config_dir() == ROOT / "configs"
    assert paths.config_file() == ROOT / "configs" / "monitor.yaml"


def test_config_dir_congelado_usa_el_directorio_del_sistema(tmp_path, monkeypatch):
    """Un binario no tiene un configs/ del repo donde escribir."""
    _congelar(monkeypatch, tmp_path / "bundle")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    if sys.platform.startswith("linux"):
        assert paths.config_dir() == (tmp_path / "xdg" / paths.APP_NAME).resolve()
    assert paths.config_dir().name == paths.APP_NAME


def test_la_variable_de_entorno_fuerza_el_directorio_de_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path))
    assert paths.config_dir() == tmp_path.resolve()


def test_la_auditoria_va_junto_a_la_configuracion(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path))
    assert paths.audit_file() == tmp_path.resolve() / "monitor_audit.jsonl"


def test_el_ejemplo_de_config_viaja_con_el_programa():
    assert paths.example_config().is_file()
    assert paths.example_config() == paths.bundle_root() / "configs" / "monitor.example.yaml"


# ── Diagnóstico ──────────────────────────────────────────────────────────────

def test_describe_expone_las_tres_raices():
    d = paths.describe()
    assert set(d) == {"frozen", "bundle_root", "data_root", "config_dir"}
    assert d["frozen"] is False


# ── Capacidades de la plataforma ─────────────────────────────────────────────

import os  # noqa: E402
import subprocess  # noqa: E402

import psutil  # noqa: E402

from monitor_api import platform_caps, poller  # noqa: E402


def test_hardware_temps_refleja_lo_que_hay():
    """En Windows psutil ni define la función; aquí sí."""
    assert platform_caps.hardware_temps_available() == hasattr(psutil, "sensors_temperatures")


def test_runner_python_desde_el_codigo_fuente_es_el_actual():
    assert platform_caps.runner_python({}) == sys.executable


def test_runner_python_congelado_no_devuelve_el_binario(tmp_path, monkeypatch):
    """Regresión: `sys.executable` congelado ES el monitor.

    Lanzarlo en vez del runner arrancaría otro monitor, en bucle.
    """
    _congelar(monkeypatch, tmp_path)
    elegido = platform_caps.runner_python({})
    assert elegido != sys.executable
    # O encuentra un python de verdad en el PATH, o admite que no puede.
    assert elegido is None or Path(elegido).name.startswith("python")


def test_runner_python_respeta_el_configurado(tmp_path):
    falso = tmp_path / "mi-python"
    falso.write_text("#!/bin/sh\n")
    assert platform_caps.runner_python({"python_executable": str(falso)}) == str(falso)


def test_runner_python_rechaza_un_configurado_inexistente(tmp_path):
    assert platform_caps.runner_python({"python_executable": str(tmp_path / "no")}) is None


def test_runner_launch_exige_los_scripts_del_pipeline(tmp_path):
    paths.set_data_root(tmp_path)
    assert platform_caps.runner_launch_available({}) is False

    (tmp_path / "scripts").mkdir()
    assert platform_caps.runner_launch_available({}) is True


def test_describe_cubre_lo_que_el_frontend_necesita():
    d = platform_caps.describe({})
    assert set(d) == {
        "os", "frozen", "hardware_temps", "runner_launch", "runner_python",
        "auto_advance",
    }


# ── Procesos multiplataforma ─────────────────────────────────────────────────

def test_grupo_propio_usa_la_primitiva_de_cada_sistema(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    kwargs = poller._grupo_propio()
    assert "creationflags" in kwargs and "start_new_session" not in kwargs

    monkeypatch.setattr(sys, "platform", "linux")
    assert poller._grupo_propio() == {"start_new_session": True}


def test_job_processes_encuentra_por_cwd():
    """Antes se recorría /proc a mano; ahora psutil, que va en Windows también."""
    pids = poller._job_processes(Path.cwd())
    assert os.getpid() in pids


def test_job_processes_no_ve_procesos_de_otro_directorio(tmp_path):
    assert os.getpid() not in poller._job_processes(tmp_path)


def test_job_processes_incluye_los_descendientes(tmp_path):
    hijo = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path)
    try:
        pids = poller._job_processes(tmp_path, root_pid=os.getpid())
        assert hijo.pid in pids, "el árbol de descendientes debe entrar"
    finally:
        hijo.kill()
        hijo.wait(timeout=5)


def test_pid_alive_y_rss_usan_psutil():
    assert poller._is_pid_alive(os.getpid()) is True
    assert poller._is_pid_alive(4_000_000) is False
    assert poller._proc_rss_mb(os.getpid()) > 0
    assert poller._proc_rss_mb(4_000_000) is None


def test_kill_job_processes_termina_el_arbol(tmp_path):
    """Se comprueba sobre procesos propios de prueba, nunca sobre jobs reales."""
    hijo = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path)
    try:
        matados = poller._kill_job_processes(tmp_path)
        assert hijo.pid in matados
        hijo.wait(timeout=5)
        assert hijo.poll() is not None, "el proceso debería haber terminado"
    finally:
        if hijo.poll() is None:
            hijo.kill()
            hijo.wait(timeout=5)


# ── Degradación honesta en los endpoints ─────────────────────────────────────

pytest.importorskip("fastapi", reason="requiere el extra [web]")
from fastapi.testclient import TestClient  # noqa: E402

from monitor_api.main import create_app  # noqa: E402


class _PollerMinimo:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.cfg: dict = {}
        self._snapshots: dict = {}
        self.last_poll_at = None

    @property
    def snapshots(self):
        return self._snapshots


def _cliente(tmp_path):
    app = create_app(config={})
    app.state.poller = _PollerMinimo(tmp_path)
    app.state.hub = None
    return TestClient(app)


def test_health_publica_rutas_y_capacidades(tmp_path):
    body = _cliente(tmp_path).get("/api/health").json()

    assert set(body["paths"]) == {"frozen", "bundle_root", "data_root", "config_dir"}
    assert body["platform"]["os"] == sys.platform
    assert isinstance(body["platform"]["hardware_temps"], bool)
    assert isinstance(body["platform"]["runner_launch"], bool)


def test_lanzar_runner_da_501_cuando_no_se_puede(tmp_path):
    """Sin scripts/ del pipeline no hay nada que lanzar.

    501 y no 500: es una capacidad ausente, no un fallo del servidor.
    """
    paths.set_data_root(tmp_path)  # tmp_path no tiene scripts/
    r = _cliente(tmp_path).post("/api/batches/0/start")

    assert r.status_code == 501
    assert "intérprete" in r.json()["detail"] or "scripts" in r.json()["detail"]


def test_lanzar_runner_no_da_501_cuando_si_se_puede(tmp_path):
    """Con la capacidad disponible, el 404 del batch inexistente vuelve a mandar."""
    (tmp_path / "scripts").mkdir()
    paths.set_data_root(tmp_path)
    r = _cliente(tmp_path).post("/api/batches/999/start")

    assert r.status_code == 404


def test_system_responde_aunque_no_haya_sensores(tmp_path, monkeypatch):
    """En Windows psutil no define sensors_temperatures; antes era un 500."""
    monkeypatch.delattr(psutil, "sensors_temperatures", raising=False)

    r = _cliente(tmp_path).get("/api/system")
    assert r.status_code == 200
    body = r.json()
    assert body["pkg_temps"] == []
    assert body["gpu_temps"] == []
    assert body["cpu_percent"] >= 0


# ── Versión única ────────────────────────────────────────────────────────────

import json  # noqa: E402
import tomllib  # noqa: E402

from monitor_api import __version__  # noqa: E402


def test_la_version_tiene_forma_de_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-.].+)?", __version__), __version__


def test_pyproject_toma_la_version_de_ahi():
    """Se declara dinámica para no tener dos números que puedan divergir."""
    datos = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in datos["project"].get("dynamic", [])
    assert datos["tool"]["hatch"]["version"]["path"] == "src/monitor_api/__init__.py"


def test_el_frontend_declara_la_misma_version():
    pkg = json.loads((ROOT / "frontend" / "package.json").read_text())
    assert pkg["version"] == __version__


def test_la_app_y_health_anuncian_la_version(tmp_path):
    cliente = _cliente(tmp_path)
    assert cliente.get("/api/health").json()["version"] == __version__


# ── Artefactos de empaquetado ────────────────────────────────────────────────

SPEC = ROOT / "packaging" / "dft-monitor-web.spec"
ENGINE_SPEC = ROOT / "packaging" / "dft-monitor-engine.spec"
ENTRY = ROOT / "packaging" / "entry.py"
BUILD = ROOT / "scripts" / "build_web.sh"
FLUTTER_BUILD = ROOT / "scripts" / "build_desktop.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def test_el_entry_point_no_usa_imports_relativos():
    """Regresión: PyInstaller ejecuta el script como `__main__` suelto.

    Un `from .launcher import main` ahí revienta con «attempted relative import
    with no known parent package», y el binario no arranca.
    """
    import ast as _ast

    assert ENTRY.is_file()
    arbol = _ast.parse(ENTRY.read_text())

    # Sobre el AST y no sobre el texto: el docstring del archivo menciona el
    # antipatrón para explicarlo, y una búsqueda de cadenas se engancharía ahí.
    relativos = [
        n for n in _ast.walk(arbol)
        if isinstance(n, _ast.ImportFrom) and (n.level or 0) > 0
    ]
    assert not relativos, "el entry point no puede usar imports relativos"

    absolutos = {
        f"{n.module}.{a.name}"
        for n in _ast.walk(arbol)
        if isinstance(n, _ast.ImportFrom) and n.module
        for a in n.names
    }
    assert "monitor_api.launcher.main" in absolutos


def test_el_spec_congela_el_entry_point_correcto():
    contenido = SPEC.read_text()
    assert 'packaging" / "entry.py' in contenido
    assert '"__main__.py"' not in contenido


def test_el_spec_excluye_lo_que_no_debe_viajar():
    contenido = SPEC.read_text()
    for modulo in ("matplotlib", "ase", "gpaw", "torch", "tkinter"):
        assert f'"{modulo}"' in contenido, f"{modulo} debería estar en excludes"


def test_el_spec_empaqueta_los_recursos_que_paths_espera():
    """`paths.bundle_root()` los busca con estos nombres exactos."""
    contenido = SPEC.read_text()
    for destino in ('"static"', '"configs"', '"structures"', '"models"'):
        assert destino in contenido


def test_el_build_es_ejecutable_y_prueba_el_binario():
    assert BUILD.is_file() and os.access(BUILD, os.X_OK)
    contenido = BUILD.read_text()
    # Un fallo de import o un recurso que no viajó solo se ve ejecutándolo.
    assert "Smoke test" in contenido or "smoke" in contenido.lower()
    assert "/api/health" in contenido
    assert "/api/ml/predict" in contenido


def test_el_build_flutter_declara_engine_embebido():
    assert ENGINE_SPEC.is_file()
    assert FLUTTER_BUILD.is_file() and os.access(FLUTTER_BUILD, os.X_OK)
    spec = ENGINE_SPEC.read_text()
    script = FLUTTER_BUILD.read_text()

    assert "dft-monitor-engine" in spec
    assert "--print-ready-json" in script
    assert "assets/engine" in script


def test_el_workflow_verifica_el_tag_contra_la_version():
    assert WORKFLOW.is_file()
    contenido = WORKFLOW.read_text()
    assert "__version__" in contenido
    assert "no coincide" in contenido
    assert "windows-latest" in contenido and "ubuntu-22.04" in contenido


def test_la_conversion_de_estructuras_sigue_existiendo():
    """Se usa en el build para que `ase` no entre en el artefacto."""
    script = ROOT / "scripts" / "pregenerate_structures.py"
    assert script.is_file()
    # El código conserva el camino de conversión para ejecutar desde el repo.
    from monitor_api.services import files as files_svc

    assert hasattr(files_svc, "_ase_json_a_cif")


def test_el_modelo_de_health_declara_todo_lo_que_describe_produce(tmp_path):
    """Pydantic descarta en silencio lo que el modelo no declara.

    Regresión: `auto_advance` se añadió a platform_caps.describe() y no a
    PlatformInfo, así que nunca llegaba al frontend.
    """
    from monitor_api import platform_caps

    body = _cliente(tmp_path).get("/api/health").json()
    assert set(body["platform"]) == set(platform_caps.describe({}))


# ── Supervisión del orquestador ──────────────────────────────────────────────

class _ProcFalso:
    """Popen mínimo: `poll()` devuelve lo que se le diga."""
    def __init__(self, codigo=None):
        self._codigo = codigo
        self.stderr = None

    def poll(self):
        return self._codigo


def _poller_desnudo():
    """Instancia sin __init__: solo queremos los métodos de bandera."""
    from monitor_api.poller import DFTPoller
    return DFTPoller.__new__(DFTPoller)


def test_orquestador_muerto_libera_la_bandera():
    """Un orquestador que revienta no debe dejar el monitor creyéndose ocupado.

    Regresión: la bandera se ponía a True al lanzar y solo se limpiaba cuando
    aparecía un batch preparado, así que un fallo la dejaba fija para siempre.
    """
    p = _poller_desnudo()
    p._orchestrator_running = True
    p._orchestrator_proc = _ProcFalso(codigo=1)

    assert p._orchestrator_vivo() is False
    assert p._orchestrator_running is False   # liberada: se puede reintentar


def test_orquestador_vivo_no_se_relanza():
    p = _poller_desnudo()
    p._orchestrator_running = True
    p._orchestrator_proc = _ProcFalso(codigo=None)

    assert p._orchestrator_vivo() is True
    assert p._orchestrator_running is True


def test_orquestador_sin_handle_se_respeta():
    """Bandera puesta sin proceso asociado: no lo damos por muerto."""
    p = _poller_desnudo()
    p._orchestrator_running = True
    p._orchestrator_proc = None

    assert p._orchestrator_vivo() is True


def test_backoff_frena_el_bucle_de_crashes():
    """Tras un fallo no se relanza de inmediato.

    Cosechar el fallo sin enfriamiento convierte el bloqueo permanente en un
    bucle: cada ciclo relanzaría el mismo orquestador contra la misma causa.
    """
    import time as _t
    from monitor_api import poller as P

    p = _poller_desnudo()
    p._orchestrator_running = True
    p._orchestrator_proc = _ProcFalso(codigo=1)
    p._orchestrator_vivo()                      # cosecha el fallo

    assert p._orchestrator_fallos == 1
    assert p._orchestrator_en_espera() is True  # recién fallado: esperar

    # Pasado el enfriamiento, se permite reintentar.
    p._orchestrator_ultimo_fallo = _t.time() - P.ORCHESTRATOR_BACKOFF_S - 1
    assert p._orchestrator_en_espera() is False


def test_backoff_crece_con_los_fallos():
    import time as _t
    from monitor_api import poller as P

    p = _poller_desnudo()
    p._orchestrator_fallos = 3
    # Justo antes del enfriamiento del tercer fallo (4× la base) todavía espera.
    p._orchestrator_ultimo_fallo = _t.time() - P.ORCHESTRATOR_BACKOFF_S * 4 + 30
    assert p._orchestrator_en_espera() is True
    p._orchestrator_ultimo_fallo = _t.time() - P.ORCHESTRATOR_BACKOFF_S * 4 - 1
    assert p._orchestrator_en_espera() is False


def test_se_rinde_tras_demasiados_fallos():
    from monitor_api import poller as P

    p = _poller_desnudo()
    p._orchestrator_fallos = P.ORCHESTRATOR_MAX_FALLOS
    p._orchestrator_ultimo_fallo = 0.0     # hace una eternidad
    assert p._orchestrator_en_espera() is True   # no reintenta jamás


def test_exito_reinicia_el_contador():
    p = _poller_desnudo()
    p._orchestrator_fallos = 3
    p._orchestrator_running = True
    p._orchestrator_proc = _ProcFalso(codigo=0)

    p._orchestrator_vivo()
    assert p._orchestrator_fallos == 0
    assert p._orchestrator_en_espera() is False


# ── Config de primer arranque ────────────────────────────────────────────────

def _primera_config(tmp_path, monkeypatch, *, frozen: bool) -> str:
    """Genera la config inicial en un directorio limpio y devuelve su texto."""
    from monitor_api import launcher, paths

    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(paths, "is_frozen", lambda: frozen)
    if frozen:
        # `bundle_root()` va a `sys._MEIPASS` en cuanto is_frozen() es cierto;
        # apuntándolo al repo, el ejemplo de configuración se encuentra igual.
        monkeypatch.setattr(sys, "_MEIPASS", str(ROOT), raising=False)
    launcher.preparar_config("127.0.0.1", announce=False)
    return (tmp_path / "cfg" / "monitor.yaml").read_text(encoding="utf-8")


def test_escritorio_no_muta_el_pipeline_al_abrir(tmp_path, monkeypatch):
    """Congelado, la config inicial deja auto_advance en false.

    Un doble clic sobre un lote terminado disparaba el orquestador de active
    learning sin preguntar.
    """
    texto = _primera_config(tmp_path, monkeypatch, frozen=True)
    assert "auto_advance: false" in texto
    assert "auto_advance: true" not in texto


def test_desde_el_repo_se_conserva_el_desatendido(tmp_path, monkeypatch):
    texto = _primera_config(tmp_path, monkeypatch, frozen=False)
    assert "auto_advance: true" in texto


def test_stderr_del_orquestador_no_es_una_tuberia(tmp_path, monkeypatch):
    """El stderr debe ir a archivo, nunca a `subprocess.PIPE`.

    Una tubería que nadie drena se llena a los 64 KB y bloquea al orquestador
    indefinidamente; el monitor lo vería «corriendo» para siempre.
    """
    fuente = (ROOT / "src" / "monitor_api" / "poller.py").read_text(encoding="utf-8")
    bloque = fuente.split("advance\"], cwd=str(raiz_datos)")[1].split(")")[0]
    assert "subprocess.PIPE" not in bloque
    assert "_orchestrator_log_fh" in bloque


def test_el_log_del_orquestador_vive_en_config(tmp_path, monkeypatch):
    from monitor_api import poller as P

    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path))
    assert P.orchestrator_log() == tmp_path / "orchestrator.log"


# ── Dependencias que el binario necesita de verdad ───────────────────────────

def test_el_spec_no_excluye_ase():
    """`ase` debe viajar en el binario.

    Regresión: se excluía para ahorrar 26 MB, asumiendo que solo servía para
    convertir cuatro JSON a CIF. Pero preparar los jobs DFT del cribado
    construye las estructuras ABX3 en proceso (`build_abx3.build` importa
    `ase.build` de forma diferida), así que
    POST /api/screening/runs/{id}/start-dft devolvía 500.
    """
    spec = (ROOT / "packaging" / "dft-monitor-engine.spec").read_text(encoding="utf-8")
    bloque = spec.split("excludes = [")[1].split("]")[0]
    assert '"ase"' not in bloque
    assert '"gpaw"' in bloque      # gpaw sí se queda fuera: corre en el Python externo


def test_el_spec_declara_los_imports_diferidos():
    """La cadena de preparación se importa dentro de funciones.

    El análisis estático de PyInstaller no la alcanza, así que tiene que estar
    declarada a mano o el binario se queda sin ella.
    """
    spec = (ROOT / "packaging" / "dft-monitor-engine.spec").read_text(encoding="utf-8")
    bloque = spec.split("hiddenimports = [")[1].split("]")[0]
    for modulo in ("ase.build", "ase.spacegroup",
                   "dft_cspbi3.structure_builder", "buho.structure.build_abx3"):
        assert f'"{modulo}"' in bloque, f"falta {modulo} en hiddenimports"


def test_un_import_ausente_no_da_500():
    """Si falta una dependencia, el usuario debe leer cuál — no un 500 opaco."""
    router = (ROOT / "src" / "monitor_api" / "router.py").read_text(encoding="utf-8")
    bloque = router.split("async def screening_start_dft")[1].split("@router")[0]
    assert "except ImportError" in bloque
    assert "501" in bloque


# ── Los dos productos no deben confundirse ───────────────────────────────────

def test_los_dos_artefactos_se_llaman_distinto():
    """`web` y `desktop` deben ser distinguibles por el nombre del archivo.

    Antes el artefacto de la web app se llamaba `dft-monitor-<v>` a secas y el
    de escritorio `dft-monitor-flutter-<v>`: el nombre neutro parecía el
    principal y «flutter» no le dice nada a quien descarga.
    """
    web = (ROOT / "scripts" / "build_web.sh").read_text(encoding="utf-8")
    esc = (ROOT / "scripts" / "build_desktop.sh").read_text(encoding="utf-8")

    assert 'NOMBRE="dft-monitor-web-${VERSION}' in web
    assert 'OUT_NAME="dft-monitor-desktop-${VERSION}' in esc
    assert "dft-monitor-flutter" not in esc


def test_las_entradas_de_menu_no_comparten_nombre():
    """Dos .desktop con `Name=Monitor DFT` eran indistinguibles en el menú."""
    esc = (ROOT / "packaging" / "dft-monitor-desktop.desktop").read_text(encoding="utf-8")
    web = (ROOT / "packaging" / "dft-monitor-web.desktop").read_text(encoding="utf-8")

    def nombre(texto: str) -> str:
        return next(l.split("=", 1)[1] for l in texto.splitlines() if l.startswith("Name="))

    assert nombre(esc) != nombre(web)
    assert "web" in nombre(web).lower()


def test_el_instalador_retira_los_nombres_antiguos():
    """Reinstalar sobre el esquema viejo dejaba los dos conviviendo."""
    inst = (ROOT / "scripts" / "install_launcher.sh").read_text(encoding="utf-8")
    assert "quitar_legado" in inst
    assert inst.count("quitar_legado") >= 3   # definición + instalar + desinstalar


def test_el_build_de_escritorio_es_reproducible_desde_cero():
    """`native_assets/<plataforma>` debe crearse antes de `flutter build`.

    Flutter 3.47 lo referencia desde su cmake_install.cmake pero no lo crea si
    el proyecto no declara assets nativos. La compilación solo funcionaba porque
    el directorio sobrevivía de veces anteriores: con un `build/` virgen —CI, o
    al mover el directorio a tmpfs— abortaba con «file INSTALL cannot find».
    """
    script = (ROOT / "scripts" / "build_desktop.sh").read_text(encoding="utf-8")
    pos_mkdir = script.find('mkdir -p "build/native_assets/$FLUTTER_TARGET"')
    pos_build = script.find('flutter build "$FLUTTER_TARGET"')
    assert pos_mkdir != -1, "falta la creación de native_assets"
    assert pos_mkdir < pos_build, "debe crearse ANTES de compilar"


# ── El agente es opcional; el monitor no ─────────────────────────────────────

def test_el_ejemplo_no_lleva_rutas_personales():
    """El ejemplo viaja dentro del binario que descarga cualquiera.

    Llevaba `revive_repo: /home/luis-ochoa/Documents/Vscode/revive-rocm-gfx803`
    con `manage_service: true`, así que un primer arranque en otra máquina
    intentaba `make ollama-serve` sobre un repositorio inexistente.
    """
    texto = (ROOT / "configs" / "monitor.example.yaml").read_text(encoding="utf-8")
    assert "/home/" not in texto
    assert "manage_service: false" in texto


def test_el_generator_distribuible_no_lleva_rutas_de_una_maquina():
    """`config/generator.dist.yaml` es lo que se empaqueta en el binario.

    Regresión: se estaba empaquetando `config/generator.yaml` a secas, con
    `/home/luis/perovowl-micromamba/...`, `/media/luis-ochoa/...`, `C:/NuevoVol`
    y `/mnt/c/Users/LUIS/...` dentro. En otra máquina el motor arrancaba pero
    el cribado y el DFT apuntaban a rutas inexistentes, con fallos confusos en
    vez de un "no configurado, corre el wizard de Entorno".
    """
    dist = ROOT / "config" / "generator.dist.yaml"
    assert dist.is_file(), "falta config/generator.dist.yaml"
    texto = dist.read_text(encoding="utf-8")
    for marca in ("/home/", "/media/", "/mnt/", "C:/Users", "C:/NuevoVol",
                  "micromamba", "luis-ochoa"):
        assert marca not in texto, f"generator.dist.yaml contiene {marca!r}"

    import yaml

    cfg = yaml.safe_load(texto)
    # Las claves que el servicio de descubrimiento y la cascada necesitan.
    assert "chemical_space" in cfg and "filters" in cfg and "discovery" in cfg
    # Sin los bloques específicos de la máquina.
    assert "wsl" not in cfg["discovery"]
    assert "windows_mounts" not in cfg["discovery"]
    assert "wsl" not in cfg["discovery"].get("mlff", {})
    # Rutas de salida relativas, no absolutas.
    for clave, valor in cfg.get("paths", {}).items():
        assert not Path(valor).is_absolute(), f"paths.{clave} es absoluta: {valor}"


def test_el_spec_empaqueta_el_generator_distribuible():
    """El spec debe empaquetar la variante `.dist`, renombrada a generator.yaml."""
    spec = ENGINE_SPEC.read_text(encoding="utf-8")
    assert "generator.dist.yaml" in spec
    # Llega al bundle como `config/generator.yaml`, que es lo que el motor busca.
    assert '"generator.yaml"' in spec
    assert "generator.yaml\"), \"config\"" not in spec  # no la de dev, a secas


def test_un_ollama_caido_no_impide_arrancar():
    """Regresión: `raise SystemExit` tumbaba el monitor entero.

    Bastaba reiniciar la máquina —o abrir el binario en otro equipo— para que
    el monitor no levantara, porque el asistente LLM opcional no respondía.
    """
    fuente = (ROOT / "src" / "monitor_api" / "launcher.py").read_text(encoding="utf-8")
    bloque = fuente.split("ensure_managed_ollama(agent_cfg")[1].split("url = f\"http")[0]
    assert "SystemExit" not in bloque
    assert "no disponible" in bloque


# ── Los procesos largos deben sobrevivir a la app ────────────────────────────

def test_ningun_hijo_hereda_la_salida_del_monitor():
    """Heredar stdout/stderr ata el hijo a la vida de quien lanzó el monitor.

    Regresión real: la app de escritorio lee el motor por tuberías. El runner
    las heredaba, y al cerrar la app moría de SIGPIPE en su siguiente `print`.
    El log de batch_765153 termina en un LAUNCH sin su DONE, con 42 jobs sin
    hacer. `start_new_session` desliga el grupo de procesos, no los
    descriptores.
    """
    import re

    for rel in ("src/monitor_api/poller.py", "src/monitor_api/services/bench.py"):
        fuente = (ROOT / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"subprocess\.Popen\(", fuente):
            # Bloque de la llamada: hasta que se equilibran los paréntesis.
            i, prof = m.end() - 1, 0
            while i < len(fuente):
                if fuente[i] == "(":
                    prof += 1
                elif fuente[i] == ")":
                    prof -= 1
                    if prof == 0:
                        break
                i += 1
            bloque = fuente[m.start():i + 1]
            linea = fuente[:m.start()].count("\n") + 1
            assert "stdout=" in bloque, f"{rel}:{linea} hereda stdout del monitor"
            assert "stderr=" in bloque, f"{rel}:{linea} hereda stderr del monitor"


def test_los_imports_perezosos_estan_declarados_en_el_spec():
    """PyInstaller analiza imports estáticos: los diferidos hay que declararlos.

    v0.7.0 salió con `buho.structure.fases` dentro del binario y sin spglib,
    porque `grupo_espacial()` lo importa dentro de la función. Identificar la
    fase —lo que esa versión anunciaba— devolvía «spglib no instalado» en todo
    binario publicado. Es el mismo patrón que dejó la tabla del scissor SOC sin
    cargar: algo que solo se nota ejecutando el paquete.
    """
    import re

    spec = (ROOT / "packaging" / "dft-monitor-engine.spec").read_text(encoding="utf-8")
    fuente = (ROOT / "src" / "buho").rglob("*.py")

    # Módulos importados dentro de una función (indentados), no al principio.
    perezosos = set()
    for py in fuente:
        for linea in py.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s+(?:import|from)\s+([a-z_][a-z0-9_]*)", linea)
            if m and linea.startswith((" ", "\t")):
                perezosos.add(m.group(1))

    # Solo se exigen los de terceros que el binario necesita de verdad.
    vigilados = {"spglib"} & perezosos
    for modulo in sorted(vigilados):
        assert f'"{modulo}"' in spec, (
            f"'{modulo}' se importa de forma perezosa en buho/ pero no está en "
            "hiddenimports: el binario saldría sin él y fallaría en silencio")
