"""Arranque rápido: sondeo de la máquina y orquestación del protocolo."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitor_api import paths  # noqa: E402
from monitor_api.services import hwprobe, quickstart  # noqa: E402


# ── Sondeo ───────────────────────────────────────────────────────────────────

def test_el_sondeo_mide_lo_que_dice_medir():
    m = hwprobe.medir(con_benchmark=False)

    assert m["nucleos_fisicos"] >= 1
    assert m["nucleos_logicos"] >= m["nucleos_fisicos"]
    assert m["ram_total_gb"] > 0
    assert 0 <= m["ram_disponible_gb"] <= m["ram_total_gb"]


def test_el_micro_benchmark_da_un_numero_positivo():
    """El recuento de núcleos miente en VMs y portátiles estrangulados."""
    b = hwprobe._micro_benchmark()
    if not b.get("disponible"):
        pytest.skip(f"benchmark no disponible: {b.get('motivo')}")
    assert b["gflops"] > 0
    assert b["repeticiones"] >= 1


def test_la_ram_limita_los_slots_aunque_sobren_nucleos():
    """Abrir más trabajos de los que caben en RAM hace paginar, y paginar es
    más caro que un slot de menos."""
    reparto = hwprobe.repartir({
        "nucleos_fisicos": 64,
        "ram_disponible_gb": 10.0,   # 8 GB utilizables -> 4 jobs de 2 GB
    })
    assert reparto["runner_slots"] == 4
    assert reparto["limitado_por"] == "ram"


def test_los_nucleos_limitan_cuando_sobra_memoria():
    reparto = hwprobe.repartir({
        "nucleos_fisicos": 4,
        "ram_disponible_gb": 256.0,
    })
    assert reparto["runner_slots"] == 3      # 4 menos uno reservado
    assert reparto["limitado_por"] == "cpu"


def test_una_maquina_pequena_recibe_un_slot_y_un_aviso():
    """Nunca cero: un trabajo aunque apriete, pero diciéndolo."""
    reparto = hwprobe.repartir({
        "nucleos_fisicos": 2,
        "ram_disponible_gb": 2.5,
    })
    assert reparto["runner_slots"] == 1
    assert reparto["runner_cores"] >= 1
    assert reparto["aviso"], "quedarse sin RAM no puede pasar en silencio"


def test_nunca_se_comprometen_todos_los_nucleos():
    """La API y la GUI también necesitan CPU."""
    for n in (2, 4, 8, 16, 44, 128):
        r = hwprobe.repartir({"nucleos_fisicos": n, "ram_disponible_gb": 1024.0})
        assert r["nucleos_en_uso"] <= n - hwprobe.NUCLEOS_RESERVADOS


def test_el_reparto_explica_que_lo_limito():
    """Dos slots en una máquina de 16 núcleos hay que poder discutirlos."""
    r = hwprobe.repartir({"nucleos_fisicos": 16, "ram_disponible_gb": 8.0})
    assert r["limitado_por"] in {"ram", "cpu"}
    assert r["supuestos"]["ram_por_job_gb"] == hwprobe.RAM_POR_JOB_GB


def test_el_sondeo_se_declara_estimacion():
    """No es la calibración: `bench_machine.py` mide t/iter con GPAW real."""
    s = hwprobe.sondear(con_benchmark=False)
    assert s["es_estimacion"] is True
    assert "bench_machine" in s["refinar_con"]


# ── Orquestación ─────────────────────────────────────────────────────────────

def test_no_arranca_si_falta_un_requisito(tmp_path, monkeypatch):
    """El bucle corre días como subproceso: si falta GPAW hay que decirlo
    antes, no cuando la primera ronda ya preparó los trabajos."""
    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))
    quickstart.reset_for_tests()

    monkeypatch.setattr(quickstart, "_faltantes", lambda: [
        {"id": "dft", "titulo": "Runtime DFT", "error": "no se encuentra gpaw",
         "remediacion": "buho setup install dft"},
    ])
    arrancado = []
    monkeypatch.setattr(
        "monitor_api.services.discovery.start",
        lambda **kw: arrancado.append(kw),
    )

    r = quickstart.arrancar(con_benchmark=False)

    assert r["ok"] is False
    assert r["motivo"] == "faltan requisitos"
    assert arrancado == [], "no puede arrancar el bucle con requisitos ausentes"
    paso = next(p for p in r["pasos"] if p["paso"] == "prerequisitos")
    assert paso["detalle"]["faltantes"][0]["remediacion"] == "buho setup install dft"


def test_el_reparto_se_escribe_en_la_configuracion(tmp_path, monkeypatch):
    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))
    quickstart.reset_for_tests()
    monkeypatch.setattr(quickstart, "_faltantes", lambda: [{"id": "x", "titulo": "x"}])

    quickstart.arrancar(con_benchmark=False)

    import yaml
    cfg = yaml.safe_load(paths.config_file().read_text(encoding="utf-8"))
    # Bajo `monitor:`: es la subseccion que el poller recibe como su cfg.
    # Al nivel superior el YAML queda valido y nadie lo lee.
    assert cfg["monitor"]["runner_slots"] >= 1
    assert cfg["monitor"]["runner_cores"] >= 1
    assert "runner_slots" not in cfg, "no puede quedar suelto en la raiz"


def test_escribir_el_reparto_no_borra_el_resto_de_la_configuracion(tmp_path, monkeypatch):
    """monitor.yaml es del usuario: lleva su token y sus rutas."""
    paths.set_data_root(tmp_path)
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir(parents=True)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(cfgdir))
    (cfgdir / "monitor.yaml").write_text(
        "monitor:\n  poll_interval_sec: 45\n  runner_slots: 1\n"
        "telegram:\n  bot_token: secreto\n", encoding="utf-8")

    quickstart._persistir_reparto({"runner_slots": 6, "runner_cores": 3})

    import yaml
    cfg = yaml.safe_load((cfgdir / "monitor.yaml").read_text(encoding="utf-8"))
    assert cfg["monitor"]["runner_slots"] == 6
    assert cfg["monitor"]["runner_cores"] == 3
    assert cfg["monitor"]["poll_interval_sec"] == 45
    assert cfg["telegram"]["bot_token"] == "secreto"


def test_arranca_el_protocolo_cuando_todo_esta_listo(tmp_path, monkeypatch):
    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))
    quickstart.reset_for_tests()
    monkeypatch.setattr(quickstart, "_faltantes", lambda: [])
    monkeypatch.setattr(
        "monitor_api.services.discovery.status",
        lambda: {"state": {"status": "not_initialized"}},
    )
    monkeypatch.setattr("monitor_api.services.discovery.init", lambda **kw: {"n": 10})
    lanzado = {}
    monkeypatch.setattr(
        "monitor_api.services.discovery.start",
        lambda **kw: lanzado.update(kw) or {"state": {"status": "running"}},
    )

    r = quickstart.arrancar(con_benchmark=False, max_rounds=3)

    assert r["ok"] is True
    assert lanzado["max_rounds"] == 3
    assert lanzado["start_runner"] is True
    assert [p["paso"] for p in r["pasos"]] == [
        "sondeo", "configuracion", "prerequisitos", "init", "protocolo",
    ]


def test_no_vuelve_a_enumerar_un_espacio_ya_creado(tmp_path, monkeypatch):
    """Reenumerar borraría el ledger con el historial de DFT."""
    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))
    quickstart.reset_for_tests()
    monkeypatch.setattr(quickstart, "_faltantes", lambda: [])
    monkeypatch.setattr(
        "monitor_api.services.discovery.status",
        lambda: {"state": {"status": "paused"}},
    )

    def _no_llamar(**kw):
        raise AssertionError("init no debe llamarse sobre un espacio existente")

    monkeypatch.setattr("monitor_api.services.discovery.init", _no_llamar)
    monkeypatch.setattr("monitor_api.services.discovery.start", lambda **kw: {})

    r = quickstart.arrancar(con_benchmark=False)

    assert r["ok"] is True
    paso = next(p for p in r["pasos"] if p["paso"] == "init")
    assert "omitido" in paso["detalle"]


def test_un_sondeo_roto_no_impide_arrancar(tmp_path, monkeypatch):
    """Medir es para ajustar, no un requisito: sin medida se usa lo configurado."""
    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))
    quickstart.reset_for_tests()

    def _revienta(**kw):
        raise RuntimeError("psutil no disponible")

    monkeypatch.setattr(hwprobe, "sondear", _revienta)
    monkeypatch.setattr(quickstart, "_faltantes", lambda: [])
    monkeypatch.setattr(
        "monitor_api.services.discovery.status",
        lambda: {"state": {"status": "not_initialized"}},
    )
    monkeypatch.setattr("monitor_api.services.discovery.init", lambda **kw: {})
    monkeypatch.setattr("monitor_api.services.discovery.start", lambda **kw: {})

    r = quickstart.arrancar()

    assert r["ok"] is True
    sondeo = next(p for p in r["pasos"] if p["paso"] == "sondeo")
    assert sondeo["ok"] is False
    assert "psutil" in sondeo["error"]


# ── API ──────────────────────────────────────────────────────────────────────

def test_el_sondeo_por_api_no_arranca_nada(tmp_path, monkeypatch):
    """Mirar cuántos trabajos saldrían no puede lanzar días de cálculo."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from monitor_api.main import create_app

    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))

    def _no_llamar(**kw):
        raise AssertionError("GET /probe no debe arrancar el protocolo")

    monkeypatch.setattr("monitor_api.services.discovery.start", _no_llamar)

    r = TestClient(create_app(config={})).get("/api/quickstart/probe")

    assert r.status_code == 200
    assert r.json()["reparto"]["runner_slots"] >= 1
    assert r.json()["es_estimacion"] is True


def test_max_rounds_invalido_se_rechaza(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from monitor_api.main import create_app

    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))
    r = TestClient(create_app(config={})).post(
        "/api/quickstart", json={"max_rounds": 0})
    assert r.status_code == 422
