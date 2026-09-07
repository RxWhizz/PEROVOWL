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
    assert wsl["setup_path"].endswith("/share/gpaw/setups")
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
