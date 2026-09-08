"""Instalación del runtime DFT (GPAW en WSL) desde la app.

Instalar WSL en sí no se intenta: `wsl --install` exige administrador y
reinicio. Lo que sí se automatiza es todo lo que viene después de que exista una
distribución, incluido escribir dónde quedó el entorno — sin eso el entorno se
crea y nadie sabe encontrarlo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buho import setup_wizard  # noqa: E402
from monitor_api import paths  # noqa: E402
from monitor_api.services import setup as setup_service  # noqa: E402


# ── Cuándo hay plan y cuándo solo instrucciones ──────────────────────────────

def test_sin_wsl_no_hay_pasos_sino_el_comando_exacto(monkeypatch):
    """Pedir elevación desde una app científica es intrusivo; se guía."""
    monkeypatch.setattr(setup_wizard.sys, "platform", "win32")
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: False)

    plan = setup_wizard.plan_dft({})

    assert plan.steps == [], "sin wsl.exe no se puede instalar nada"
    assert any("wsl --install" in n for n in plan.notas)
    assert any("administrador" in n for n in plan.notas)


def test_con_wsl_pero_sin_distro_se_propone_una(monkeypatch):
    """La primera vez pide usuario y contraseña interactivamente."""
    monkeypatch.setattr(setup_wizard.sys, "platform", "win32")
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: [])

    plan = setup_wizard.plan_dft({})

    assert plan.steps == []
    assert any(setup_wizard.DISTRO_SUGERIDA in n for n in plan.notas)


def test_se_reutiliza_la_distro_que_ya_existe(monkeypatch):
    """Instalar otra distro a alguien que ya tiene la suya sería invasivo."""
    monkeypatch.setattr(setup_wizard.sys, "platform", "win32")
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: ["Debian", "Ubuntu"])

    plan = setup_wizard.plan_dft({})

    assert plan.steps, "con distro disponible sí hay plan"
    argv = plan.steps[0].argv
    assert argv[:3] == ["wsl.exe", "-d", "Debian"]


def test_fuera_de_windows_no_aplica(monkeypatch):
    monkeypatch.setattr(setup_wizard.sys, "platform", "linux")
    plan = setup_wizard.plan_dft({})
    assert plan.steps == []
    assert any("WSL" in n for n in plan.notas)


# ── El plan ──────────────────────────────────────────────────────────────────

@pytest.fixture
def plan_listo(monkeypatch):
    monkeypatch.setattr(setup_wizard.sys, "platform", "win32")
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: ["Ubuntu"])
    return setup_wizard.plan_dft({})


def test_el_plan_crea_entorno_y_descarga_datasets(plan_listo):
    nombres = [s.name for s in plan_listo.steps]
    assert nombres == [
        "asegurar-micromamba", "crear-entorno", "datasets-paw", "verificar",
    ]


def test_numpy_queda_fijado_para_no_romper_gpaw(plan_listo):
    """GPAW 24.6 no compila contra la ABI de numpy 2. Es la razón de que el
    entorno MLFF viva aparte en vez de compartir este."""
    crear = next(s for s in plan_listo.steps if s.name == "crear-entorno")
    script = " ".join(crear.argv)
    assert f"numpy={setup_wizard.GPAW_NUMPY}" in script
    assert setup_wizard.GPAW_NUMPY.startswith("1.")


def test_la_version_de_gpaw_esta_fijada(plan_listo):
    """«La última» ha roto el runner más de una vez."""
    crear = next(s for s in plan_listo.steps if s.name == "crear-entorno")
    assert f"gpaw={setup_wizard.GPAW_VERSION}" in " ".join(crear.argv)


def test_recrear_limpia_antes_y_es_opcional(monkeypatch):
    """Un `env remove` que falla porque no había nada no invalida el plan."""
    monkeypatch.setattr(setup_wizard.sys, "platform", "win32")
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: ["Ubuntu"])

    plan = setup_wizard.plan_dft({}, recrear=True)

    limpiar = next(s for s in plan.steps if s.name == "limpiar")
    assert limpiar.opcional is True
    assert plan.steps.index(limpiar) < [s.name for s in plan.steps].index("crear-entorno")


def test_el_plan_avisa_de_lo_que_va_a_descargar(plan_listo):
    assert any("GB" in n for n in plan_listo.notas)


def test_se_reutiliza_micromamba_del_entorno_existente():
    """Quien ya tiene el entorno MLFF no debe bajar micromamba otra vez."""
    cfg = {"discovery": {"wsl": {
        "python": "/home/u/perovowl-micromamba/envs/gpaw246/bin/python"}}}
    rutas = setup_wizard._rutas_gpaw(cfg, "gpaw246")
    assert rutas["micromamba"] == "/home/u/perovowl-micromamba/bin/micromamba"


def test_en_maquina_limpia_micromamba_va_a_una_ruta_absoluta():
    """Un «micromamba» pelado dejaría el binario en el directorio actual."""
    rutas = setup_wizard._rutas_gpaw({}, "gpaw246")
    assert rutas["micromamba"].startswith("$HOME/")


# ── La traducción de rutas, que es lo que hace usable el entorno ─────────────

def test_la_raiz_de_datos_se_traduce_a_ruta_de_wsl(tmp_path, monkeypatch):
    """Ahí es donde `materializar_pipeline` deja scripts/ y src/, y de ahí los
    importa el Python de WSL. Sin la traducción el entorno existe pero no
    encuentra el pipeline."""
    paths.set_data_root(tmp_path)
    monkeypatch.setattr(
        paths, "data_root", lambda: Path("C:/Users/quien/PEROVOWL-data"))

    assert setup_service._raiz_datos_en_wsl() == "/mnt/c/Users/quien/PEROVOWL-data"


def test_una_raiz_sin_letra_de_unidad_no_se_inventa_una(monkeypatch):
    monkeypatch.setattr(paths, "data_root", lambda: Path("/home/x/datos"))
    assert setup_service._raiz_datos_en_wsl() is None


# ── Escribir dónde quedó el entorno ──────────────────────────────────────────

def test_al_instalar_se_escribe_donde_quedo_el_entorno(tmp_path, monkeypatch):
    """Sin esto el entorno se crea y `setup_wizard` sigue diciendo
    «Define discovery.wsl.python»."""
    paths.set_data_root(tmp_path)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: ["Ubuntu"])
    monkeypatch.setattr(setup_service, "_config", lambda: {})

    r = setup_service.configurar_wsl_dft()

    assert r["escrito"] is True
    cfg = yaml.safe_load(
        (tmp_path / "config" / "generator.yaml").read_text(encoding="utf-8"))
    wsl = cfg["discovery"]["wsl"]
    assert wsl["distro"] == "Ubuntu"
    assert wsl["python"].endswith("/envs/gpaw246/bin/python")
    assert wsl["setup_path"].endswith("/site-packages/gpaw_data/setups")
    assert wsl["project_root"].startswith("/mnt/")


def test_escribir_la_config_conserva_el_espacio_quimico(tmp_path, monkeypatch):
    """generator.yaml lleva las puertas del cribado y la química del usuario."""
    paths.set_data_root(tmp_path)
    destino = tmp_path / "config"
    destino.mkdir(parents=True)
    (destino / "generator.yaml").write_text(
        "chemical_space:\n  A_sites: [Cs, Rb]\n"
        "generation:\n  fractions: [0.5]\n", encoding="utf-8")
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: ["Ubuntu"])
    monkeypatch.setattr(setup_service, "_config", lambda: {})

    setup_service.configurar_wsl_dft()

    cfg = yaml.safe_load((destino / "generator.yaml").read_text(encoding="utf-8"))
    assert cfg["chemical_space"]["A_sites"] == ["Cs", "Rb"]
    assert cfg["generation"]["fractions"] == [0.5]
    assert cfg["discovery"]["wsl"]["distro"] == "Ubuntu"


def test_sin_distros_no_se_escribe_nada(tmp_path, monkeypatch):
    paths.set_data_root(tmp_path)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: [])

    r = setup_service.configurar_wsl_dft()

    assert r["escrito"] is False
    assert not (tmp_path / "config" / "generator.yaml").exists()


def test_dft_esta_en_el_despachador():
    """Antes solo se podía instalar `mlff` y los grupos pip."""
    plan = setup_wizard.plan("dft", config={})
    assert plan.target == "dft"


def test_los_setups_apuntan_a_donde_conda_los_deja():
    """`gpaw-data` de conda-forge los deja en site-packages, no en share/gpaw.

    Comprobado sobre un entorno real: `share/gpaw` no existe. Apuntar ahí
    dejaba `setup_path` señalando a un directorio vacío y GPAW sin datasets.
    """
    rutas = setup_wizard._rutas_gpaw({}, "gpaw246")
    assert rutas["setups"].endswith("/site-packages/gpaw_data/setups")
    assert "/share/gpaw" not in rutas["setups"]


def test_los_datasets_no_se_descargan_si_ya_estan(plan_listo):
    """`gpaw-data` entra como dependencia: bajarlos siempre era gastar red."""
    paso = next(s for s in plan_listo.steps if s.name == "datasets-paw")
    script = paso.argv[-1]
    assert script.startswith("test -f "), "primero se comprueba"
    assert "Cs.PBE.gz" in script, "se mira un elemento concreto, no solo el directorio"
    assert "||" in script and "install-data" in script, "y solo si falta se descarga"


# ── Detección de un GPAW ya instalado ────────────────────────────────────────

def test_no_se_usan_variables_de_shell_en_la_deteccion():
    """Al pasar un script por `wsl.exe -- bash -c`, las variables propias del
    script llegan **vacías** —incluso entre comillas simples— mientras que las
    del entorno como `$HOME` sí sobreviven.

    Un `for p in ...; do ... "$p" ...; done` itera bien pero con la variable
    vacía, y la detección devolvía «no encontrado» en una máquina donde GPAW sí
    estaba. Por eso la iteración se hace en Python.
    """
    import inspect

    fuente = inspect.getsource(setup_wizard.detectar_gpaw_wsl)
    # `$HOME` sí puede aparecer: viene del entorno de WSL, no del script.
    sospechosas = [l for l in fuente.splitlines()
                   if "$" in l and "$HOME" not in l and not l.strip().startswith("#")]
    assert not sospechosas, (
        f"variables de shell en el script: {sospechosas}")


def test_sin_wsl_no_se_detecta_nada(monkeypatch):
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: False)
    assert setup_wizard.detectar_gpaw_wsl() is None


def test_se_elige_el_entorno_que_importa_gpaw(monkeypatch):
    """Existir el binario no basta: un entorno a medio crear tiene el ejecutable
    y no el paquete. Aquí hay dos y solo uno sirve."""
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)

    class _Proc:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def _falso(distro, script, *, timeout=60):
        if script.startswith("ls -d"):
            return _Proc(0, "/h/envs/sin_gpaw/bin/python\n/h/envs/con_gpaw/bin/python\n")
        if "sin_gpaw" in script:
            return _Proc(1)
        return _Proc(0, "24.6.0 3.29.0\n")

    monkeypatch.setattr(setup_wizard, "_run_wsl", _falso)
    r = setup_wizard.detectar_gpaw_wsl()

    assert r is not None
    assert r["python"] == "/h/envs/con_gpaw/bin/python"
    assert r["gpaw"] == "24.6.0"
    assert r["setup_path"].endswith("/site-packages/gpaw_data/setups")


def test_si_no_hay_ninguno_se_dice_en_vez_de_inventar(monkeypatch):
    """Sin GPAW instalado hay que instalarlo, y eso son ~2.5 GB: no se hace a
    escondidas desde el arranque rápido."""
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)

    class _Proc:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    monkeypatch.setattr(setup_wizard, "_run_wsl",
                        lambda d, s, **k: _Proc(0, "") if s.startswith("ls -d") else _Proc(1))
    assert setup_wizard.detectar_gpaw_wsl() is None


def test_el_arranque_rapido_autoconfigura_antes_de_rendirse(tmp_path, monkeypatch):
    """El caso de la captura: WSL está, GPAW está, y solo faltaba la ruta."""
    from monitor_api.services import quickstart

    paths.set_data_root(tmp_path)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: ["Ubuntu"])
    monkeypatch.setattr(setup_service, "_config", lambda: {})
    monkeypatch.setattr(
        setup_wizard, "sondear_gpaw_wsl",
        lambda *a, **k: {"wsl": True, "distros": ["Ubuntu"],
                         "candidatos": ["/h/envs/g/bin/python"],
                         "probados": [], "motivo": None,
                         "hallado": {
                             "python": "/h/envs/g/bin/python", "gpaw": "24.6.0",
                             "ase": "3.29.0", "prefix": "/h/envs/g",
                             "mpirun": "/h/envs/g/bin/mpiexec", "distro": "Ubuntu",
                             "setup_path": "/h/envs/g/lib/python3.12/"
                                           "site-packages/gpaw_data/setups"}})

    r = quickstart.autoconfigurar_dft()

    assert r["configurado"] is True
    assert r["detectado"]["gpaw"] == "24.6.0"

    # Y lo que se escribe es la ruta ENCONTRADA. Sintetizar aqui la canonica
    # (`$HOME/perovowl-micromamba/envs/gpaw246/...`) dejaria la config apuntando
    # a un directorio inexistente en cuanto el GPAW de la maquina viva en otro
    # sitio: peor que no configurar nada, porque ya no se nota que falta.
    wsl = yaml.safe_load(
        (tmp_path / "config" / "generator.yaml").read_text(encoding="utf-8")
    )["discovery"]["wsl"]
    assert wsl["python"] == "/h/envs/g/bin/python"
    assert wsl["setup_path"] == ("/h/envs/g/lib/python3.12/"
                                 "site-packages/gpaw_data/setups")
    assert wsl["env_name"] == "g"


def test_tras_instalar_se_sigue_escribiendo_la_ruta_del_instalador(tmp_path, monkeypatch):
    """Sin `rutas` la funcion deduce donde deja el entorno el instalador. Es lo
    correcto justo despues de instalar, y es el otro camino: que uno gane no
    puede romper el otro."""
    paths.set_data_root(tmp_path)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: ["Ubuntu"])
    monkeypatch.setattr(setup_service, "_config", lambda: {})

    r = setup_service.configurar_wsl_dft()

    assert r["escrito"] is True
    wsl = yaml.safe_load(
        (tmp_path / "config" / "generator.yaml").read_text(encoding="utf-8")
    )["discovery"]["wsl"]
    assert wsl["env_name"] == setup_wizard.GPAW_ENV
    assert wsl["python"].endswith(f"/envs/{setup_wizard.GPAW_ENV}/bin/python")
    assert wsl["micromamba"].endswith("/bin/micromamba")


def test_si_no_hay_nada_que_detectar_no_se_configura(tmp_path, monkeypatch):
    from monitor_api.services import quickstart

    paths.set_data_root(tmp_path)
    monkeypatch.setattr(
        setup_wizard, "sondear_gpaw_wsl",
        lambda *a, **k: {"wsl": True, "distros": ["Ubuntu"],
                         "candidatos": ["/usr/bin/python3"],
                         "probados": [{"python": "/usr/bin/python3", "gpaw": None,
                                       "error": "No module named gpaw"}],
                         "hallado": None,
                         "motivo": "se probaron 1 interpretes y ninguno importa GPAW"})

    r = quickstart.autoconfigurar_dft()

    assert r["configurado"] is False
    assert "ninguno importa GPAW" in r["motivo"]


# ── Decir lo que se vio, no repetir el consejo de otro problema ──────────────

def _finge_dft_ausente(monkeypatch):
    """La maquina de desarrollo SI tiene GPAW configurado, asi que
    `setup.status` diria que el runtime esta bien y la entrada ni aparece. Lo
    que se prueba aqui es que el mensaje cuente lo que vio el sondeo."""
    monkeypatch.setattr(setup_service, "status", lambda **k: {"capacidades": [
        {"id": "dft", "titulo": "Runtime DFT (GPAW en WSL)", "ok": False,
         "requerido": True, "error": "No hay intérprete GPAW configurado.",
         "remediacion": "Define discovery.wsl.python en config/generator.yaml."},
    ]})


def test_tras_buscar_el_mensaje_deja_de_mandar_a_editar_el_yaml(tmp_path, monkeypatch):
    """La captura de la otra maquina: la busqueda corrio, no encontro nada, y el
    mensaje seguia siendo «Define discovery.wsl.python en config/generator.yaml»
    --- un consejo para el problema contrario, porque ahi lo que falta es
    instalar 2.5 GB, no editar una linea."""
    from monitor_api.services import quickstart

    paths.set_data_root(tmp_path)
    _finge_dft_ausente(monkeypatch)
    monkeypatch.setattr(
        setup_wizard, "sondear_gpaw_wsl",
        lambda *a, **k: {"wsl": True, "distros": ["Ubuntu"],
                         "candidatos": ["/usr/bin/python3"],
                         "probados": [], "hallado": None, "motivo": "ninguno"})

    faltan = quickstart._faltantes()

    dft = [f for f in faltan if f["id"] == "dft"]
    assert dft, "el runtime DFT deberia seguir faltando"
    assert "discovery.wsl.python" not in (dft[0]["remediacion"] or "")
    assert "no esta" in dft[0]["error"] or "no está" in dft[0]["error"]
    assert "Entorno" in dft[0]["remediacion"]
    # Y el sondeo viaja con el fallo: sin el, «no lo encontre» y «no mire» se
    # ven igual desde fuera.
    assert dft[0]["sondeo"]["candidatos"] == ["/usr/bin/python3"]


def test_sin_wsl_el_mensaje_lo_dice(tmp_path, monkeypatch):
    from monitor_api.services import quickstart

    paths.set_data_root(tmp_path)
    _finge_dft_ausente(monkeypatch)
    monkeypatch.setattr(
        setup_wizard, "sondear_gpaw_wsl",
        lambda *a, **k: {"wsl": False, "candidatos": [], "probados": [],
                         "hallado": None, "motivo": "no hay wsl.exe"})

    dft = [f for f in quickstart._faltantes() if f["id"] == "dft"][0]

    assert "WSL" in dft["error"]
    assert "wsl --install" in dft["remediacion"]


def test_wsl_sin_ninguna_distro_no_se_confunde_con_wsl_sin_gpaw(tmp_path, monkeypatch):
    """`wsl.exe` viene de serie en Windows 11 aunque no haya nada dentro. Los
    dos arreglos no se parecen: `wsl --install -d Ubuntu` frente a bajar el
    runtime desde Entorno."""
    from monitor_api.services import quickstart

    paths.set_data_root(tmp_path)
    _finge_dft_ausente(monkeypatch)
    monkeypatch.setattr(
        setup_wizard, "sondear_gpaw_wsl",
        lambda *a, **k: {"wsl": True, "distros": [], "candidatos": [],
                         "probados": [], "hallado": None,
                         "motivo": "hay wsl.exe pero ninguna distro instalada"})

    dft = [f for f in quickstart._faltantes() if f["id"] == "dft"][0]

    assert "wsl --install -d Ubuntu" in dft["remediacion"]
    assert "distribución de Linux" in dft["error"]


def test_el_sondeo_mira_si_hay_distros_antes_de_buscar_interpretes(monkeypatch):
    """Sin distro, `wsl.exe -- bash` falla y la lista de candidatos sale vacía:
    indistinguible de «hay Ubuntu pero sin entornos». Por eso se comprueba
    antes."""
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard, "distros_wsl", lambda: [])
    llamadas = []
    monkeypatch.setattr(setup_wizard, "_run_wsl",
                        lambda *a, **k: llamadas.append(a) or None)

    s = setup_wizard.sondear_gpaw_wsl()

    assert s["hallado"] is None
    assert s["distros"] == []
    assert "ninguna distro" in s["motivo"]
    assert not llamadas, "no hay a quien preguntarle: no se lanza ningun bash"


def test_una_frase_de_error_no_se_toma_por_el_nombre_de_una_distro(monkeypatch):
    """Sin distros, algunas versiones de wsl.exe no fallan: imprimen una frase y
    salen con codigo 0. Tomarla por un nombre lleva a `wsl -d "Windows Subsystem
    for Linux has no installed distributions."`, que falla de una forma que no
    se parece al problema real --- que es que no hay WSL."""
    class _Proc:
        returncode = 0
        stdout = "Windows Subsystem for Linux has no installed distributions.\n".encode("utf-16-le")

    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard.subprocess, "run", lambda *a, **k: _Proc())

    assert setup_wizard.distros_wsl() == []


def test_los_nombres_de_distro_normales_siguen_pasando(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "Ubuntu\nDebian\nkali-linux\nUbuntu-22.04\n".encode("utf-16-le")

    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard.subprocess, "run", lambda *a, **k: _Proc())

    assert setup_wizard.distros_wsl() == ["Ubuntu", "Debian", "kali-linux", "Ubuntu-22.04"]


# ── Una Ubuntu recien instalada no trae bzip2 ────────────────────────────────

def test_micromamba_se_baja_sin_tar_ni_bzip2():
    """La via oficial (`curl ... | tar -xvj`) descomprime bzip2, y una Ubuntu
    recien instalada NO lo trae. En una maquina real el paso murio con
    «tar (grandchild): bzip2: Cannot exec: No such file or directory» y todo lo
    demas cayo detras por no existir micromamba."""
    script = setup_wizard._script_micromamba("/h/bin/micromamba")

    assert "tar" not in script, "sin tar: no se puede contar con bzip2"
    assert "bzip2" not in script
    assert setup_wizard.URL_MICROMAMBA in script
    assert "chmod +x" in script
    # Idempotente: no se vuelve a bajar si ya esta.
    assert script.startswith("test -x ")


def test_los_dos_planes_usan_el_mismo_arranque_de_micromamba():
    """Estaba copiado en el plan de DFT y en el de MLFF, y solo se arreglo uno
    la primera vez. Una sola fuente para que no puedan divergir."""
    import inspect

    fuente = inspect.getsource(setup_wizard)
    assert fuente.count("_script_micromamba(mm)") == 2
    assert fuente.count("micro.mamba.pm/api/micromamba") == 0
