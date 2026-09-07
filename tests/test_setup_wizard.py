"""Tests del runtime MLFF y del wizard de entorno.

Ninguno toca WSL ni instala nada: lo que se comprueba es que el plan que se
construye es el correcto y que la ausencia del entorno MLFF degrada la cascada
en vez de matarla.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ── mlff_runtime: resolucion ──────────────────────────────────────────────────


def _cfg(**mlff) -> dict:
    return {
        "discovery": {
            "wsl": {
                "distro": "Ubuntu",
                "project_root": "/mnt/c/repo",
                "python": "/opt/mm/envs/gpaw246/bin/python",
            },
            "mlff": mlff,
        }
    }


def test_resolve_usa_wsl_en_windows_por_defecto(monkeypatch):
    from buho import mlff_runtime

    monkeypatch.setattr(sys, "platform", "win32")
    rt = mlff_runtime.resolve(_cfg(wsl={"python": "/opt/mm/envs/perovowl-mlff/bin/python"}))

    assert rt.backend == "wsl"
    assert rt.python == "/opt/mm/envs/perovowl-mlff/bin/python"
    # La raiz del repo y la distro se heredan del bloque WSL del runner DFT:
    # configurarlas dos veces para la misma maquina seria pedir un error.
    assert rt.distro == "Ubuntu"
    assert rt.worker == "/mnt/c/repo/scripts/buho_mlff_worker.py"


def test_resolve_backend_off_no_ejecuta_nada():
    from buho import mlff_runtime

    rt = mlff_runtime.resolve(_cfg(backend="off"))
    assert rt.backend == "off"
    with pytest.raises(mlff_runtime.MLFFUnavailableError):
        rt.command(["--preflight-only"])


def test_env_var_gana_a_la_config(monkeypatch):
    from buho import mlff_runtime

    monkeypatch.setenv("BUHO_MLFF_BACKEND", "local")
    monkeypatch.setenv("BUHO_MLFF_PYTHON", "/usr/bin/python3")
    rt = mlff_runtime.resolve(_cfg(backend="wsl"))

    assert rt.backend == "local"
    assert rt.python == "/usr/bin/python3"


def test_backend_no_reconocido_es_error_explicito():
    from buho import mlff_runtime

    with pytest.raises(mlff_runtime.MLFFUnavailableError):
        mlff_runtime.resolve(_cfg(backend="quantum"))


def test_command_wsl_incluye_distro_y_cd():
    from buho import mlff_runtime

    rt = mlff_runtime.MLFFRuntime(
        backend="wsl", python="/opt/py", worker="/mnt/c/repo/w.py",
        distro="Ubuntu", project_root="/mnt/c/repo",
    )
    cmd = rt.command(["--preflight-only"])

    assert cmd[:4] == ["wsl.exe", "-d", "Ubuntu", "--"]
    assert cmd[4] == "bash"
    assert "cd /mnt/c/repo" in cmd[-1]
    assert "--preflight-only" in cmd[-1]


def test_command_local_es_argv_directo():
    from buho import mlff_runtime

    rt = mlff_runtime.MLFFRuntime(backend="local", python="/usr/bin/python3", worker="/w.py")
    assert rt.command(["--stdin"]) == ["/usr/bin/python3", "/w.py", "--stdin"]


# ── cascada: degradacion ──────────────────────────────────────────────────────


def _cascade_cfg() -> dict:
    return {
        "random_seed": 42,
        "chemical_space": {"A_sites": ["Cs", "Rb"], "B_sites": ["Pb", "Sn"],
                           "X_sites": ["I", "Br"]},
        "filters": {
            "goldschmidt": {"min": 0.0, "max": 10.0},
            "octahedral": {"min": 0.0, "max": 10.0},
            "volume_A3": {"min": 0.0, "max": 10000.0},
        },
        "structure": {
            "supercell_pure": [1, 1, 1],
            "supercell_mixed": [1, 1, 1],
            "organic_A_placeholder": "Cs",
        },
        "screening": {"tier1_surrogate": False, "tier2_mlff": True,
                      "tier1_gate": False, "tier2_gate": True},
        "acquisition": {"beta": 1.0, "pv_window": [1.1, 1.8]},
        "discovery": {"mlff": {"backend": "off"}},
    }


def _candidatos(n: int = 3):
    from buho.discovery.space import ChemicalSpaceEnumerator

    cfg = _cascade_cfg()
    cfg["generation"] = {
        "fractions": [0.5], "fraction_mode": "discrete",
        "modes": {"pure": True, "A_mixed": False, "B_mixed": False,
                  "X_mixed": False, "multi_mixed": False},
    }
    cfg["discovery"]["space"] = {"min_fraction": 0.5, "max_fraction": 0.5,
                                 "fraction_step": 0.5, "include_multi_mixed": False}
    candidatos, _ = ChemicalSpaceEnumerator(cfg).enumerate(physical_viable_only=True)
    assert len(candidatos) >= n, f"el espacio de prueba solo dio {len(candidatos)} candidatos"
    return candidatos[:n]


def test_tier2_sin_entorno_lanza_error_tipado():
    """Falta del entorno != fallo de un candidato: tiene que ser distinguible."""
    from buho.mlff_runtime import MLFFUnavailableError
    from buho.screening.cascade import ScreeningCascade

    cascade = ScreeningCascade(_cascade_cfg(), project_root=ROOT)
    with pytest.raises(MLFFUnavailableError):
        cascade.screen(_candidatos(), run_mlff=True)


def test_tier2_desactivado_no_impide_tier0(monkeypatch):
    from buho.screening.cascade import ScreeningCascade

    cascade = ScreeningCascade(_cascade_cfg(), project_root=ROOT)
    df = cascade.screen(_candidatos(), run_mlff=False)

    assert not df.empty
    assert set(df["tier_reached"]) <= {0, 1}


class _FakeRuntime:
    """Runtime MLFF que responde sin lanzar procesos."""

    backend = "wsl"

    def __init__(self, respuestas):
        self._respuestas = respuestas
        self.llamadas = []

    def predict(self, candidates, config, timeout=None):
        self.llamadas.append(list(candidates))
        return self._respuestas


def test_tier2_remoto_manda_un_solo_lote_y_mapea_por_id():
    """Cargar MEGNet cuesta mas que predecir: un proceso por lote, no por candidato."""
    from buho.screening.cascade import ScreeningCascade

    candidatos = _candidatos(3)
    respuestas = [
        {"candidate_id": c.candidate_id, "Eg_gnn_eV": 1.4,
         "Eform_megnet_eV_atom": -0.5, "Eform_m3gnet_eV_atom": -0.3,
         "structure_source": "cubic", "error": None}
        for c in candidatos
    ]
    fake = _FakeRuntime(respuestas)
    cascade = ScreeningCascade(_cascade_cfg(), project_root=ROOT, mlff_runtime=fake)

    df = cascade.screen(candidatos, run_mlff=True)

    assert len(fake.llamadas) == 1
    assert len(fake.llamadas[0]) == len(candidatos)
    evaluados = df[df["tier_reached"] == 2]
    assert len(evaluados) == len(candidatos)
    # Eform combinado = media de los dos modelos; std = mitad de la discrepancia.
    assert evaluados["Eform_eV_atom"].iloc[0] == pytest.approx(-0.4)
    assert evaluados["Eform_std_eV_atom"].iloc[0] == pytest.approx(0.1)


def test_tier2_remoto_error_por_candidato_no_tumba_el_lote():
    from buho.screening.cascade import ScreeningCascade

    candidatos = _candidatos(2)
    respuestas = [
        {"candidate_id": candidatos[0].candidate_id, "Eg_gnn_eV": 1.4,
         "Eform_megnet_eV_atom": -0.5, "Eform_m3gnet_eV_atom": -0.5,
         "structure_source": "cubic", "error": None},
        {"candidate_id": candidatos[1].candidate_id,
         "error": "no se pudo construir la estructura: boom"},
    ]
    cascade = ScreeningCascade(_cascade_cfg(), project_root=ROOT,
                               mlff_runtime=_FakeRuntime(respuestas))
    df = cascade.screen(candidatos, run_mlff=True)

    caido = df[df["candidate_id"] == candidatos[1].candidate_id].iloc[0]
    assert caido["dropped_at_tier"] == 2
    assert "boom" in caido["drop_reason"]
    # El otro sobrevive con su Eform: un fallo aislado no invalida el lote.
    vivo = df[df["candidate_id"] == candidatos[0].candidate_id].iloc[0]
    assert vivo["Eform_eV_atom"] == pytest.approx(-0.5)


# ── engine: no dejar el estado atascado ───────────────────────────────────────


def _engine_config(tmp_path: Path) -> Path:
    cfg = {
        "version": "test",
        "random_seed": 42,
        "chemical_space": {"A_sites": ["Cs", "MA"], "B_sites": ["Pb", "Sn"],
                           "X_sites": ["I", "Br"]},
        "generation": {
            "fractions": [0.5], "fraction_mode": "continuous",
            "n_samples_per_combo": 1, "batch_size": 20,
            "modes": {"pure": True, "A_mixed": True, "B_mixed": True,
                      "X_mixed": True, "multi_mixed": False},
        },
        "filters": {
            "goldschmidt": {"min": 0.0, "max": 10.0},
            "octahedral": {"min": 0.0, "max": 10.0},
            "volume_A3": {"min": 0.0, "max": 10000.0},
        },
        "screening": {"tier1_surrogate": False, "tier2_mlff": True,
                      "tier1_gate": False, "tier2_gate": False, "n_dft_per_batch": 0},
        "acquisition": {"beta": 1.0, "pv_window": [1.1, 1.8]},
        "discovery": {
            "output_dir": "data/discovery", "dft_per_round": 3, "min_pv_score": 0.0,
            "frontier_size": 20, "pareto_input_size": 20, "mlff_pool_size": 10,
            "require_mlff_for_dft": True,
            # Sin entorno MLFF: es justo el caso que reventaba la ronda.
            "mlff": {"backend": "off"},
            "space": {"fraction_step": 0.5, "min_fraction": 0.5,
                      "max_fraction": 0.5, "include_multi_mixed": False},
        },
        "paths": {"runs_batches_dir": "runs/batches"},
    }
    path = tmp_path / "generator.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_score_space_degrada_a_tier01_sin_entorno_mlff(tmp_path):
    """El bug original: sin torch, la ronda moria y el estado quedaba en 'screening'."""
    from buho.discovery import DiscoveryLoop

    loop = DiscoveryLoop(config_path=_engine_config(tmp_path), project_root=ROOT,
                         data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)

    screen = loop.score_space()

    assert screen["mlff_warning"] is not None
    assert screen["n_mlff"] == 0
    # Hay frontera pese a no tener Tier 2: require_mlff_for_dft cede cuando el
    # entorno no existe, en vez de dejar 0 elegibles y declarar "done".
    assert screen["n_frontier"] > 0

    estado = loop._load_state()
    assert estado["status"] == "idle"
    assert estado.get("mlff_warning")


def test_score_space_marca_error_en_vez_de_quedarse_en_screening(tmp_path, monkeypatch):
    """Cualquier fallo inesperado debe dejar el estado diagnosticable, no colgado."""
    from buho.discovery import DiscoveryLoop

    loop = DiscoveryLoop(config_path=_engine_config(tmp_path), project_root=ROOT,
                         data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)

    def _boom(*args, **kwargs):
        raise ValueError("fallo inesperado")

    monkeypatch.setattr(loop, "_load_candidates", _boom)

    with pytest.raises(ValueError):
        loop.score_space()

    estado = loop._load_state()
    assert estado["status"] == "error"
    assert "fallo inesperado" in estado["last_error"]


# ── wizard ────────────────────────────────────────────────────────────────────


def test_plan_mlff_crea_entorno_separado_de_gpaw():
    """Compartir env con gpaw246 romperia GPAW: numpy 1.26 contra numpy>=2."""
    from buho import setup_wizard

    plan = setup_wizard.plan_mlff(_cfg(wsl={
        "env_name": "perovowl-mlff",
        "micromamba": "/opt/mm/bin/micromamba",
        "distro": "Ubuntu",
        "project_root": "/mnt/c/repo",
    }))

    comandos = " ".join(s.shell() for s in plan.steps)
    assert "perovowl-mlff" in comandos
    assert "gpaw246" not in comandos
    assert [s.name for s in plan.steps] == [
        "asegurar-micromamba", "crear-entorno", "instalar-torch",
        "instalar-mlff", "verificar",
    ]


def test_plan_mlff_usa_la_rueda_cpu_salvo_que_pidas_cuda():
    from buho import setup_wizard

    base = _cfg(wsl={"micromamba": "/opt/mm/bin/micromamba", "project_root": "/mnt/c/repo"})

    cpu = " ".join(s.shell() for s in setup_wizard.plan_mlff(base).steps)
    assert setup_wizard.TORCH_CPU_INDEX in cpu

    cuda = " ".join(s.shell() for s in setup_wizard.plan_mlff(base, cuda=True).steps)
    assert setup_wizard.TORCH_CPU_INDEX not in cuda


def test_plan_mlff_recrear_borra_antes_y_es_opcional():
    from buho import setup_wizard

    plan = setup_wizard.plan_mlff(
        _cfg(wsl={"micromamba": "/opt/mm/bin/micromamba", "project_root": "/mnt/c/repo"}),
        recrear=True,
    )
    # `limpiar` va tras asegurar micromamba: sin el binario no se puede borrar.
    limpiar = next(s for s in plan.steps if s.name == "limpiar")
    assert plan.steps.index(limpiar) == 1
    # Que el entorno no exista todavia no puede abortar la creacion.
    assert limpiar.opcional


def test_plan_deduce_micromamba_del_runtime_gpaw():
    """La raiz de micromamba sale de donde ya vive el python de GPAW."""
    from buho import setup_wizard

    plan = setup_wizard.plan_mlff({"discovery": {
        "wsl": {"python": "/home/u/mm/envs/gpaw246/bin/python", "project_root": "/mnt/c/repo"},
    }})
    assert "/home/u/mm/bin/micromamba" in plan.steps[0].shell()


def test_plan_pip_se_niega_en_binario_congelado(monkeypatch):
    from buho import setup_wizard

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    plan = setup_wizard.plan_pip("web")

    assert plan.steps == []
    assert any("congelado" in n for n in plan.notas)


def test_plan_objetivo_desconocido():
    from buho import setup_wizard

    with pytest.raises(ValueError):
        setup_wizard.plan("no-existe")


def test_check_devuelve_matriz_de_capacidades():
    from buho import setup_wizard

    data = setup_wizard.check({}, project_root=ROOT, incluir_mlff=False)

    assert data["status"] in {"ok", "degradado", "error"}
    ids = {c["id"] for c in data["capacidades"]}
    assert {"core", "web", "dft", "paw"} <= ids
    for cap in data["capacidades"]:
        assert {"id", "titulo", "ok", "requerido", "remediacion"} <= set(cap)


# ── API ───────────────────────────────────────────────────────────────────────


def test_endpoints_setup_responden(monkeypatch):
    pytest.importorskip("fastapi", reason="requiere el extra [web]")
    from fastapi.testclient import TestClient

    from monitor_api.main import create_app
    from monitor_api.services import setup as service

    estado = {
        "status": "degradado", "ok": True, "plataforma": "win32",
        "python": "3.12.10", "executable": "py.exe", "frozen": False,
        "capacidades": [{"id": "mlff", "titulo": "MLFF", "ok": False,
                         "requerido": False, "detalle": {}, "error": "falta",
                         "remediacion": "instala", "comando": "buho setup install mlff"}],
        "job": {"running": False, "log": []},
    }
    llamadas: dict = {}

    monkeypatch.setattr(service, "status", lambda fast=False: {**estado, "fast": fast})
    monkeypatch.setattr(service, "job", lambda: {"running": False, "log": []})
    monkeypatch.setattr(service, "plan", lambda target, **kw: {"target": target, "steps": [], "notas": []})
    def _install(target, **kw):
        llamadas["install"] = (target, kw)
        return {"running": True, "log": []}

    monkeypatch.setattr(service, "start_install", _install)

    client = TestClient(create_app())

    assert client.get("/api/setup/status").json()["status"] == "degradado"
    # El flag `fast` tiene que llegar al servicio: es lo que evita sondear WSL.
    assert client.get("/api/setup/status", params={"fast": True}).json()["fast"] is True
    assert client.get("/api/setup/job").json()["running"] is False
    assert client.post("/api/setup/plan", json={"target": "mlff"}).json()["target"] == "mlff"

    r = client.post("/api/setup/install", json={"target": "mlff", "recreate": True})
    assert r.status_code == 200
    target, kw = llamadas["install"]
    assert target == "mlff"
    assert kw["recrear"] is True


def test_install_en_curso_responde_409(monkeypatch):
    pytest.importorskip("fastapi", reason="requiere el extra [web]")
    from fastapi.testclient import TestClient

    from monitor_api.main import create_app
    from monitor_api.services import setup as service

    def _ocupado(target, **kw):
        raise RuntimeError("Ya hay una instalación en curso.")

    monkeypatch.setattr(service, "start_install", _ocupado)
    r = TestClient(create_app()).post("/api/setup/install", json={"target": "mlff"})

    # 409, no 500: es una colisión de estado, no un fallo del servidor.
    assert r.status_code == 409
    assert "en curso" in r.json()["detail"]


def test_opciones_solo_se_pasan_a_mlff():
    from monitor_api.router import SetupInstallRequest, _setup_opciones

    assert _setup_opciones(SetupInstallRequest(target="web", cuda=True)) == {}
    opts = _setup_opciones(SetupInstallRequest(target="mlff", cuda=True))
    assert opts["cuda"] is True


# ── Troceado del lote ─────────────────────────────────────────────────────────


def test_predict_trocea_el_lote(monkeypatch):
    """5000 candidatos en una llamada son 5 MB de stdin y 7 min sin señales."""
    from buho import mlff_runtime

    rt = mlff_runtime.MLFFRuntime(backend="local", python="/py", worker="/w.py",
                                  chunk_size=2)
    trozos: list[int] = []

    # MLFFRuntime es un dataclass frozen: se parchea la clase, no la instancia.
    def _run(self, args, stdin=None, timeout=None):
        import json as _json
        enviados = _json.loads(stdin)["candidates"]
        trozos.append(len(enviados))
        return {"results": [{"candidate_id": c["candidate_id"]} for c in enviados]}

    monkeypatch.setattr(mlff_runtime.MLFFRuntime, "_run", _run)
    cands = [{"candidate_id": str(i)} for i in range(5)]
    out = rt.predict(cands, {})

    assert trozos == [2, 2, 1]
    # Trocear no puede perder ni reordenar resultados.
    assert [r["candidate_id"] for r in out] == ["0", "1", "2", "3", "4"]


def test_predict_lote_vacio_no_lanza_proceso(monkeypatch):
    from buho import mlff_runtime

    rt = mlff_runtime.MLFFRuntime(backend="local", python="/py", worker="/w.py")
    monkeypatch.setattr(mlff_runtime.MLFFRuntime, "_run",
                        lambda *a, **k: pytest.fail("no debía ejecutarse"))

    assert rt.predict([], {}) == []


def test_chunk_size_sale_de_la_config():
    from buho import mlff_runtime

    rt = mlff_runtime.resolve(_cfg(backend="local", chunk_size=250))
    assert rt.chunk_size == 250
    assert mlff_runtime.resolve(_cfg(backend="local")).chunk_size == mlff_runtime.DEFAULT_CHUNK


def test_estado_atascado_en_screening_se_recupera_solo(tmp_path):
    """El caso real: la ronda murió por falta de torch y dejó el estado colgado.

    `advance()` no trata "screening" como un estado en curso, así que vuelve a
    cribar; lo que importa es que ahora esa criba termina en vez de repetir el
    mismo fallo, y que el estado deja de estar colgado sin tocar nada a mano.
    """
    from buho.discovery import DiscoveryLoop

    loop = DiscoveryLoop(config_path=_engine_config(tmp_path), project_root=ROOT,
                         data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)

    # Se reproduce el estado que dejó el fallo del 2026-09-03.
    estado = loop._load_state()
    estado["status"] = "screening"
    estado["current_round"] = 1
    loop._save_state(estado)

    loop.score_space()

    recuperado = loop._load_state()
    assert recuperado["status"] == "idle"
    assert recuperado.get("mlff_warning")


def test_versiones_siempre_es_un_mapa(monkeypatch):
    """La GUI itera `versiones`; un string se pintaría como un chip por letra."""
    from buho import setup_wizard

    class _Proc:
        returncode = 0
        stdout = "24.6.0 3.29.0\n"
        stderr = ""

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard, "_run_wsl", lambda *a, **k: _Proc())

    cap = setup_wizard._check_dft(_cfg())

    assert cap["ok"]
    assert cap["detalle"]["versiones"] == {"gpaw": "24.6.0", "ase": "3.29.0"}


def test_versiones_vacio_si_wsl_no_responde(monkeypatch):
    from buho import setup_wizard

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(setup_wizard, "_wsl_disponible", lambda: True)
    monkeypatch.setattr(setup_wizard, "_run_wsl", lambda *a, **k: None)

    cap = setup_wizard._check_dft(_cfg())

    assert not cap["ok"]
    assert cap["detalle"]["versiones"] == {}


# ── Bucle como subproceso ─────────────────────────────────────────────────────


def test_comando_bucle_relanza_el_binario_si_esta_congelado(monkeypatch, tmp_path):
    """Congelado no hay `python -m` al que llamar: sys.executable ES el binario."""
    from monitor_api import paths
    from monitor_api.services import discovery as svc

    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "executable", r"C:\app\dft-monitor-engine.exe")

    argv = svc._comando_bucle(start_runner=True, dry_run=False, use_mlff=None, max_rounds=None)

    assert argv[0] == r"C:\app\dft-monitor-engine.exe"
    assert "-m" not in argv
    assert "--discovery-loop" in argv


def test_comando_bucle_desde_fuentes_usa_modulo(monkeypatch, tmp_path):
    from monitor_api import paths, platform_caps
    from monitor_api.services import discovery as svc

    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(platform_caps, "runner_python", lambda cfg=None: "/usr/bin/python3")

    argv = svc._comando_bucle(start_runner=False, dry_run=True, use_mlff=False, max_rounds=2)

    assert argv[:3] == ["/usr/bin/python3", "-m", "monitor_api"]
    for flag in ("--discovery-loop", "--no-runner", "--dry-run", "--no-mlff"):
        assert flag in argv
    assert argv[argv.index("--max-rounds") + 1] == "2"


def test_start_no_lanza_dos_bucles(monkeypatch, tmp_path):
    from monitor_api import paths
    from monitor_api.services import discovery as svc

    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "resolve_data", lambda rel: tmp_path / rel)
    (tmp_path / "data" / "discovery").mkdir(parents=True, exist_ok=True)
    svc._pid_file().write_text(json.dumps({"pid": 4242}), encoding="utf-8")
    monkeypatch.setattr(svc, "_vivo", lambda pid: True)
    monkeypatch.setattr(svc, "status", lambda: {"ya": "corriendo"})

    def _no(*a, **k):
        pytest.fail("no debía lanzarse un segundo bucle")

    monkeypatch.setattr(svc.subprocess, "Popen", _no)
    assert svc.start() == {"ya": "corriendo"}


def test_background_reporta_el_fallo_del_subproceso(monkeypatch, tmp_path):
    """Muerto el subproceso, su log es lo único que explica por qué."""
    from monitor_api import paths
    from monitor_api.services import discovery as svc

    monkeypatch.setattr(paths, "resolve_data", lambda rel: tmp_path / rel)
    (tmp_path / "data" / "discovery").mkdir(parents=True, exist_ok=True)
    svc._pid_file().write_text(json.dumps({"pid": 999}), encoding="utf-8")
    svc._log_file().write_text("arrancando\nTraceback (most recent call last):\nboom\n",
                               encoding="utf-8")
    monkeypatch.setattr(svc, "_vivo", lambda pid: False)

    bg = svc._background()
    assert bg["running"] is False
    assert "boom" in bg["last_error"]


def test_background_no_inventa_error_en_parada_limpia(monkeypatch, tmp_path):
    from monitor_api import paths
    from monitor_api.services import discovery as svc

    monkeypatch.setattr(paths, "resolve_data", lambda rel: tmp_path / rel)
    (tmp_path / "data" / "discovery").mkdir(parents=True, exist_ok=True)
    svc._pid_file().write_text(json.dumps({"pid": 9, "expected_exit": True}), encoding="utf-8")
    svc._log_file().write_text("Traceback\n", encoding="utf-8")
    monkeypatch.setattr(svc, "_vivo", lambda pid: False)

    assert svc._background()["last_error"] is None


# ── Metrica honesta del reentrenamiento ───────────────────────────────────────


def test_cv_detecta_un_modelo_que_no_aprende():
    """Con y aleatorio, el CV no debe batir a predecir la media."""
    import numpy as np

    from buho.discovery.engine import _cv_metrics

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))  # noqa: N806 - X matriz / y vector
    y = rng.normal(size=60)          # sin relacion con X
    cols = [f"f{i}" for i in range(5)]

    m = _cv_metrics(X, y, cols)
    assert m["cv_mae_eV"] is not None
    # El train MAE sera minusculo por memorizacion; el CV no puede mejorar la
    # linea base de forma apreciable. Eso es justo lo que el train no revelaba.
    assert m["cv_mae_eV"] >= m["baseline_mae_eV"] * 0.75


def test_cv_premia_un_modelo_que_si_aprende():
    import numpy as np

    from buho.discovery.engine import _cv_metrics

    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))  # noqa: N806 - X matriz / y vector
    y = 2.0 * X[:, 0] - X[:, 1]      # relacion clara
    cols = [f"f{i}" for i in range(5)]

    m = _cv_metrics(X, y, cols)
    assert m["cv_mae_eV"] < m["baseline_mae_eV"] * 0.6


def test_cv_se_salta_con_pocas_muestras():
    import numpy as np

    from buho.discovery.engine import _cv_metrics

    m = _cv_metrics(np.zeros((6, 5)), np.zeros(6), [f"f{i}" for i in range(5)])
    assert m["cv_mae_eV"] is None
    assert "6" in m["cv_skipped_reason"]


# ── Wizard: micromamba ────────────────────────────────────────────────────────


def test_plan_asegura_micromamba_antes_de_crear_el_entorno():
    """En una maquina limpia el plan fallaba con un 'command not found' mudo."""
    from buho import setup_wizard

    plan = setup_wizard.plan_mlff({"discovery": {"wsl": {"project_root": "/mnt/c/repo"}}})

    assert plan.steps[0].name == "asegurar-micromamba"
    cmd = plan.steps[0].shell()
    assert "micro.mamba.pm" in cmd
    # Idempotente: si ya esta, el test -x corta y no se descarga nada.
    assert "test -x" in cmd
    # Y la ruta es absoluta, no relativa al directorio actual.
    assert "$HOME/perovowl-micromamba/bin/micromamba" in cmd


def test_sh_respeta_home_pero_cita_lo_demas():
    from buho.setup_wizard import _sh

    assert _sh("$HOME/perovowl-micromamba/bin/micromamba") == "$HOME/perovowl-micromamba/bin/micromamba"
    # Nada mas se deja sin comillas: lo que venga de la config se cita.
    assert _sh("/opt/mm/bin/micromamba") == "/opt/mm/bin/micromamba"
    assert "'" in _sh("/opt/x y/mm")
    assert "'" in _sh("$HOME/../../etc/passwd; rm -rf /")


# ── El aprendizaje activo tiene que cerrar el ciclo ───────────────────────────


def test_la_cascada_prefiere_el_modelo_reentrenado(tmp_path):
    """El bucle reentrenaba cada ronda y seguia cribando con el de fabrica.

    Sin esto el "aprendizaje activo" no aprendia: proponia la misma quimica
    ronda tras ronda mientras el DFT la desmentia.
    """
    from buho.screening.cascade import ScreeningCascade

    (tmp_path / "models" / "discovery").mkdir(parents=True)
    base = tmp_path.joinpath(*ScreeningCascade.SURROGATE_BASE)
    actual = tmp_path.joinpath(*ScreeningCascade.SURROGATE_ACTUAL)

    cargados: list[Path] = []

    class _Falso:
        feature_cols = ["a"]

        @staticmethod
        def load(p):
            cargados.append(Path(p))
            return _Falso()

    import ml_surrogate.model as mm

    original = mm.SurrogateEnsemble
    mm.SurrogateEnsemble = _Falso
    try:
        # Solo el de fabrica: se usa como respaldo.
        base.write_bytes(b"x")
        c = ScreeningCascade({"screening": {"tier1_surrogate": True}}, project_root=tmp_path)
        c._load_surrogate()
        assert cargados[-1] == base

        # Con el reentrenado publicado, gana ese.
        actual.write_bytes(b"y")
        c2 = ScreeningCascade({"screening": {"tier1_surrogate": True}}, project_root=tmp_path)
        c2._load_surrogate()
        assert cargados[-1] == actual
    finally:
        mm.SurrogateEnsemble = original


def test_un_modelo_publicado_corrupto_cae_al_de_fabrica(tmp_path):
    from buho.screening.cascade import ScreeningCascade

    (tmp_path / "models" / "discovery").mkdir(parents=True)
    tmp_path.joinpath(*ScreeningCascade.SURROGATE_BASE).write_bytes(b"ok")
    tmp_path.joinpath(*ScreeningCascade.SURROGATE_ACTUAL).write_bytes(b"roto")

    intentos: list[Path] = []

    class _Falso:
        feature_cols = ["a"]

        @staticmethod
        def load(p):
            intentos.append(Path(p))
            if Path(p).name.endswith("current.pkl"):
                raise ValueError("pickle corrupto")
            return _Falso()

    import ml_surrogate.model as mm

    original = mm.SurrogateEnsemble
    mm.SurrogateEnsemble = _Falso
    try:
        c = ScreeningCascade({"screening": {"tier1_surrogate": True}}, project_root=tmp_path)
        with pytest.warns(UserWarning):
            assert c._load_surrogate() is not None
    finally:
        mm.SurrogateEnsemble = original
    # Probo el publicado y, al fallar, siguio con el de fabrica.
    assert [p.name for p in intentos] == [
        "surrogate_bandgap_current.pkl", "surrogate_bandgap.pkl",
    ]


def test_reset_de_tests_no_borra_un_bucle_vivo(monkeypatch, tmp_path):
    """La suite, corriendo contra la raiz real, borro el registro de un bucle
    en marcha: seguia cribando pero la API lo daba por muerto."""
    from monitor_api import paths
    from monitor_api.services import discovery as svc

    monkeypatch.setattr(paths, "resolve_data", lambda rel: tmp_path / rel)
    (tmp_path / "data" / "discovery").mkdir(parents=True, exist_ok=True)
    svc._pid_file().write_text(json.dumps({"pid": 4242}), encoding="utf-8")

    monkeypatch.setattr(svc, "_vivo", lambda pid: True)
    svc.reset_background_for_tests()
    assert svc._pid_file().is_file(), "no debia borrar el registro de un proceso vivo"

    monkeypatch.setattr(svc, "_vivo", lambda pid: False)
    svc.reset_background_for_tests()
    assert not svc._pid_file().is_file()


# ── El bucle no se cuelga si el runner muere a medias ─────────────────────────


def _round_runs(tmp_path: Path, round_id: int = 0):
    """Crea el directorio de runs de una ronda con jobs en estados dados."""
    d = tmp_path / "runs" / "batches" / "discovery" / f"round_{round_id:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job(runs_dir: Path, cid: str, status: str, pid: int | None = None):
    jd = runs_dir / cid
    jd.mkdir(exist_ok=True)
    payload = {"status": status, "candidate_id": cid}
    if pid is not None:
        payload["pid"] = pid
    (jd / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    return jd


def test_runner_stale_cuando_no_avanza_pese_a_un_running_fantasma(tmp_path, monkeypatch):
    """El bug real: runner WSL muerto tras 15/30, 1 job 'running' con PID muerto,
    y la deteccion antigua no marcaba stale porque no *todos* seguian pending."""
    from buho.discovery import DiscoveryLoop

    loop = DiscoveryLoop(config_path=_engine_config(tmp_path), project_root=ROOT,
                         data_root=tmp_path, models_root=tmp_path)
    runs = _round_runs(tmp_path, 0)
    for i in range(15):
        _job(runs, f"c{i}", "converged")
    for i in range(15, 29):
        _job(runs, f"c{i}", "pending")
    _job(runs, "c29", "running", pid=99999)          # PID muerto
    (runs / "runner.out").write_text("preflight ok\n", encoding="utf-8")

    # Envejecer todo lo que sirve de senal de vida.
    viejo = time.time() - 3600
    for p in list(runs.glob("**/*")):
        os.utime(p, (viejo, viejo))
    os.utime(runs / "runner.out", (viejo, viejo))

    diag = loop.runner_diagnostics(0, state={"status": "dft_running"})
    assert diag["stale"] is True
    assert diag["no_progress"] is True
    assert diag["unfinished"] == 15


def test_runner_no_stale_si_progresa(tmp_path):
    from buho.discovery import DiscoveryLoop

    loop = DiscoveryLoop(config_path=_engine_config(tmp_path), project_root=ROOT,
                         data_root=tmp_path, models_root=tmp_path)
    runs = _round_runs(tmp_path, 0)
    for i in range(10):
        _job(runs, f"c{i}", "converged")
    for i in range(10, 20):
        _job(runs, f"c{i}", "pending")
    _job(runs, "c20", "running", pid=1)
    (runs / "runner.out").write_text("STATUS ...\n", encoding="utf-8")   # recién tocado

    diag = loop.runner_diagnostics(0, state={"status": "dft_running"})
    assert diag["stale"] is False
    assert diag["no_progress"] is False


def test_reset_phantom_running_desbloquea_round_finished(tmp_path):
    from buho.discovery import DiscoveryLoop

    loop = DiscoveryLoop(config_path=_engine_config(tmp_path), project_root=ROOT,
                         data_root=tmp_path, models_root=tmp_path)
    runs = _round_runs(tmp_path, 0)
    for i in range(29):
        _job(runs, f"c{i}", "converged")
    _job(runs, "c29", "running", pid=99999)

    assert loop._round_finished(0) is False          # el 'running' lo bloquea
    n = loop._reset_phantom_running(0)
    assert n == 1
    payload = json.loads((runs / "c29" / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pending"
    assert "pid" not in payload


def test_advance_se_rinde_con_error_tras_demasiados_relanzamientos(tmp_path, monkeypatch):
    from buho.discovery import DiscoveryLoop

    loop = DiscoveryLoop(config_path=_engine_config(tmp_path), project_root=ROOT,
                         data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)
    runs = _round_runs(tmp_path, 0)
    for i in range(10):
        _job(runs, f"c{i}", "converged")
    for i in range(10, 30):
        _job(runs, f"c{i}", "pending")
    (runs / "runner.out").write_text("x\n", encoding="utf-8")
    viejo = time.time() - 3600
    for p in list(runs.glob("**/*")):
        os.utime(p, (viejo, viejo))

    st = loop._load_state()
    st["status"] = "dft_running"
    st["current_round"] = 0
    st["stale_relaunches"] = 5           # ya en el limite
    st["stale_relaunch_finished"] = 10   # el relanzamiento previo no aporto nada
    loop._save_state(st)

    # El relanzamiento real no importa: monkeypatch para no tocar WSL.
    monkeypatch.setattr(loop, "_launch_runner", lambda *a, **k: {"pid": 1})
    loop.advance(start_runner=True)

    assert loop._load_state()["status"] == "error"
    assert "no progresa" in loop._load_state()["last_error"]


# ── Radios ionicos: coordinacion del sitio A ──────────────────────────────────


def test_radio_sitio_a_es_coordinacion_12():
    """El sitio A esta rodeado por 12 aniones X; el radio tiene que ser CN12.

    Hasta 2026-09 se usaban los de CN6, etiquetados como CN12, lo que
    subestimaba el factor de tolerancia de todo el espacio de busqueda.
    Valores de Shannon 1976 (Acta Cryst A32:751).
    """
    from ml_surrogate.features import IONIC_RADII, IONIC_RADII_A_CN6

    assert IONIC_RADII["Cs"] == 1.88
    assert IONIC_RADII["Rb"] == 1.72
    assert IONIC_RADII["K"] == 1.64
    # Los de CN6 quedan disponibles, pero fuera de la tabla activa.
    assert IONIC_RADII_A_CN6 == {"Cs": 1.67, "Rb": 1.52, "K": 1.38}
    for cation, r6 in IONIC_RADII_A_CN6.items():
        assert IONIC_RADII[cation] > r6, "CN12 tiene que ser mayor que CN6"


def test_los_organicos_no_se_tocan():
    """MA/FA son radios efectivos de Kieslich, no de Shannon: no tienen CN6/CN12."""
    from ml_surrogate.features import IONIC_RADII

    assert IONIC_RADII["MA"] == 2.17
    assert IONIC_RADII["FA"] == 2.53


def test_sitios_b_y_x_siguen_en_coordinacion_6():
    """B esta en un octaedro BX6 y X entre dos octaedros: CN6 es lo correcto."""
    from ml_surrogate.features import IONIC_RADII

    assert IONIC_RADII["Pb"] == 1.19
    assert IONIC_RADII["Sn"] == 1.18
    assert IONIC_RADII["I"] == 2.20
    assert IONIC_RADII["Br"] == 1.96
    assert IONIC_RADII["Cl"] == 1.81


def test_rb_y_k_entran_al_cribado_con_el_radio_correcto():
    """La familia Rb/K con Pb/Sn estaba excluida por t<0.80 con el radio de CN6."""
    from ml_surrogate.features import IONIC_RADII, goldschmidt

    for a_site in ("Rb", "K"):
        for b_site in ("Pb", "Sn"):
            t = goldschmidt(IONIC_RADII[a_site], IONIC_RADII[b_site], IONIC_RADII["I"])
            assert 0.80 <= t <= 1.10, (
                f"{a_site}{b_site}I3 deberia pasar el filtro, da t={t:.4f}"
            )


def test_estructural_y_features_coinciden_en_cn12():
    """Dos modulos con la misma tabla no pueden discrepar sobre la coordinacion."""
    from dft_cspbi3.analysis.structural import IONIC_RADII as ESTRUCTURAL
    from ml_surrogate.features import IONIC_RADII as FEATURES

    for cation in ("Cs", "Rb", "K"):
        assert ESTRUCTURAL[cation]["CN12"] == FEATURES[cation], (
            f"{cation}: structural.py dice {ESTRUCTURAL[cation]['CN12']}, "
            f"features.py usa {FEATURES[cation]}"
        )


# ── Scissor SOC y geometria de la celda ───────────────────────────────────────


def test_scissor_depende_del_elemento_b():
    """Un chi unico para toda la familia sesgaria la comparacion entre elementos."""
    from buho.bandgap_scissor import chi_soc

    tabla = {"Pb": -0.6302, "Sn": -0.0607, "Ge": -0.2205}
    chi_pb = chi_soc({"Pb": 1.0}, tabla)
    chi_sn = chi_soc({"Sn": 1.0}, tabla)
    # El SOC crece con el numero atomico: Pb (Z=82) mucho mas que Sn (Z=50).
    assert chi_pb < chi_sn < 0
    assert abs(chi_pb - chi_sn) > 0.5, "la diferencia entre elementos no es despreciable"


def test_scissor_interpola_sitio_b_mezclado():
    from buho.bandgap_scissor import chi_soc

    tabla = {"Pb": -0.60, "Sn": -0.10}
    assert chi_soc({"Pb": 0.5, "Sn": 0.5}, tabla) == pytest.approx(-0.35)


def test_scissor_no_inventa_elementos_sin_calibrar():
    """Sin valor medido aporta 0: corregir de menos antes que adivinar."""
    from buho.bandgap_scissor import chi_soc

    assert chi_soc({"Bi": 1.0}, {"Pb": -0.63}) == 0.0


def test_scissor_no_recorta_gaps_negativos():
    """Un gap corregido negativo significa metalico; esconderlo seria mentir."""
    from buho.bandgap_scissor import corregir

    assert corregir(0.2, {"Pb": 1.0}, {"Pb": -0.63}) < 0
    assert corregir(None, {"Pb": 1.0}, {"Pb": -0.63}) is None


def test_contraccion_reproduce_las_redes_experimentales():
    """a = 2(r_B+r_X) sobreestima el enlace B-X ~9%; la contraccion lo corrige.

    El factor depende de la pareja B-X, no solo de B: calibrado con yoduros y
    aplicado a los bromuros comprimia la celda un 2.1 %, suficiente para cerrar
    el gap e invertir el orden Cl > Br > I.
    """
    from buho.structure.build_abx3 import BOND_CONTRACTION
    from ml_surrogate.features import IONIC_RADII as R

    experimental = {
        ("Pb", "I"): 6.18, ("Pb", "Br"): 5.87, ("Pb", "Cl"): 5.605,
        ("Sn", "I"): 6.22, ("Sn", "Br"): 5.80,
    }
    for (b_site, x_site), a_exp in experimental.items():
        factor = BOND_CONTRACTION[b_site][x_site]
        a = 2.0 * (R[b_site] + R[x_site]) * factor
        assert abs(a / a_exp - 1.0) < 0.01, (
            f"Cs{b_site}{x_site}3: a={a:.3f} vs exp {a_exp}")


def test_ge_no_usa_el_factor_de_pb_ni_sn():
    """Con el factor de Pb/Sn, CsGeI3 sale metalico: peor que el error original.

    Ge tiene su propio factor y va al reves --- expande, no contrae---: el radio
    ionico de Ge2+ supone un cation esferico, pero su par solitario 4s empuja
    los haluros mas lejos. Lo que no puede pasar es que herede la contraccion de
    los otros dos.
    """
    from buho.structure.build_abx3 import BOND_CONTRACTION

    ge = BOND_CONTRACTION["Ge"]
    for x_site, factor in ge.items():
        assert factor > 1.0, f"Ge-{x_site}: {factor} contraeria la celda"
        assert factor != BOND_CONTRACTION["Pb"][x_site]
        assert factor != BOND_CONTRACTION["Sn"][x_site]


# ── Riesgo de politipo: el filtro geometrico no confirma la fase ──────────────


def _filtro(**gold):
    from buho.filters.physical_filters import PhysicalFilter

    return PhysicalFilter({"filters": {"goldschmidt": gold}})


def test_cspbi3_sale_marcado_como_marginal():
    """Pasa el filtro (t=0.851) pero su fase real a 25 C es la delta."""
    from ml_surrogate.features import IONIC_RADII as R
    from ml_surrogate.features import goldschmidt

    t = goldschmidt(R["Cs"], R["Pb"], R["I"])
    f = _filtro()
    assert f.t_min <= t <= f.t_max, "sigue pasando el filtro duro"
    aviso = f.riesgo_politipo(t)
    assert aviso is not None and "fonones" in aviso


def test_zona_segura_no_marca_nada():
    f = _filtro()
    assert f.riesgo_politipo(0.95) is None


def test_riesgo_avisa_por_los_dos_lados():
    f = _filtro()
    assert "por debajo" in f.riesgo_politipo(0.82)
    assert "por encima" in f.riesgo_politipo(1.05)


def test_el_riesgo_no_rechaza_candidatos():
    """Es una etiqueta, no un filtro: no puede reducir el espacio de busqueda."""
    from buho.generator.heuristic_generator import GeneratedCandidate

    f = _filtro()
    # t marginal pero dentro del rango duro -> el check de Goldschmidt pasa.
    c = GeneratedCandidate.__new__(GeneratedCandidate)
    object.__setattr__(c, "tolerance_t", 0.82) if hasattr(
        GeneratedCandidate, "__dataclass_fields__") else setattr(c, "tolerance_t", 0.82)
    ok, _ = f._check_goldschmidt(c)
    assert ok
    assert f.riesgo_politipo(0.82) is not None


# ── Dónde se escriben los modelos reentrenados ────────────────────────────────


def test_los_modelos_reentrenados_van_a_los_datos_no_al_bundle(tmp_path, monkeypatch):
    """Regresión: `models_root` era `find_resource("models").parent`.

    Congelado eso es el directorio de extracción del binario, así que el
    surrogate reentrenado acababa dentro de la instalación: se perdía al
    actualizar y fallaba si la app estaba en un sitio de solo lectura.
    """
    from monitor_api import paths
    from monitor_api.services import discovery as svc

    datos = tmp_path / "datos"
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    datos.mkdir()

    monkeypatch.setattr(paths, "data_root", lambda: datos)
    monkeypatch.setattr(paths, "bundle_root", lambda: bundle)
    monkeypatch.setattr(paths, "find_resource", lambda *p: bundle.joinpath(*p))
    monkeypatch.setattr(svc, "_effective_config", lambda update=None: {})
    monkeypatch.setattr(svc, "config_path", lambda: tmp_path / "generator.yaml")

    loop = svc.build_loop()

    assert loop.models_root == datos, "se escribe en los datos del usuario"
    assert loop.bundle_root == bundle, "el bundle solo se lee"


def test_la_cascada_busca_modelos_en_las_dos_raices(tmp_path):
    """El de fábrica viaja en el bundle; el reentrenado vive en los datos."""
    from buho.screening.cascade import ScreeningCascade

    datos = tmp_path / "datos"
    bundle = tmp_path / "bundle"
    (datos / "models" / "discovery").mkdir(parents=True)
    bundle.joinpath(*ScreeningCascade.SURROGATE_BASE).parent.mkdir(parents=True)
    bundle.joinpath(*ScreeningCascade.SURROGATE_BASE).write_bytes(b"fabrica")

    cargados: list[Path] = []

    class _Falso:
        feature_cols = ["a"]

        @staticmethod
        def load(p):
            cargados.append(Path(p))
            return _Falso()

    import ml_surrogate.model as mm

    original = mm.SurrogateEnsemble
    mm.SurrogateEnsemble = _Falso
    try:
        cfg = {"screening": {"tier1_surrogate": True}}
        # Solo el de fábrica, en el bundle: se encuentra desde la raíz extra.
        c = ScreeningCascade(cfg, project_root=datos, extra_roots=(bundle,))
        c._load_surrogate()
        assert cargados[-1] == bundle.joinpath(*ScreeningCascade.SURROGATE_BASE)

        # Con un reentrenado en los datos, ese gana.
        actual = datos.joinpath(*ScreeningCascade.SURROGATE_ACTUAL)
        actual.write_bytes(b"reentrenado")
        c2 = ScreeningCascade(cfg, project_root=datos, extra_roots=(bundle,))
        c2._load_surrogate()
        assert cargados[-1] == actual
    finally:
        mm.SurrogateEnsemble = original


def test_la_cascada_congelada_exige_raiz_explicita(monkeypatch):
    """Sin raíz, el CWD del binario es System32: mejor fallar aquí."""
    from buho.screening.cascade import ScreeningCascade

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(ValueError, match="project_root"):
        ScreeningCascade({"screening": {}})


# ── El fallo tiene que anunciarse ─────────────────────────────────────────────


def test_los_scores_por_defecto_quedan_marcados():
    """0.5 no es neutro: es optimista para un material malo.

    Sin la marca, un score imputado es indistinguible de uno medido.
    """
    import pandas as pd

    from buho.discovery.engine import DiscoveryLoop

    loop = DiscoveryLoop.__new__(DiscoveryLoop)
    df = pd.DataFrame([
        # Con predicciones completas.
        {"candidate_id": "a", "band_score": 0.8, "stab_score": 0.7, "Eg_sigma_eV": 0.1,
         "meff_e_pred_m0": 0.3, "meff_h_pred_m0": 0.4, "eps_inf_pred": 5.0},
        # Sin ninguna: los tres terminos caen al valor por defecto.
        {"candidate_id": "b", "band_score": 0.8, "stab_score": 0.7, "Eg_sigma_eV": 0.1,
         "meff_e_pred_m0": None, "meff_h_pred_m0": None, "eps_inf_pred": None},
    ])
    out = loop._score_objectives(df).set_index("candidate_id")

    # El que tiene predicciones no lleva marca; el que no, las lleva todas.
    assert out.loc["a", "scores_imputados"] == ""
    marcados = out.loc["b", "scores_imputados"]
    for termino in ("transporte", "dielectrico", "exciton"):
        assert termino in marcados

    # Sin datos no se puede calcular la energia de enlace del exciton.
    assert pd.isna(out.loc["b", "exciton_binding_meV"])

    # Lo que hace peligroso el 0.5: "b" no tiene NINGUNA propiedad medida y
    # aun asi puntua a menos del 2 % de "a", que las tiene todas. El score no
    # delata la diferencia de informacion; solo la columna de marcas lo hace.
    a, b = out.loc["a", "pv_score_ml"], out.loc["b", "pv_score_ml"]
    assert abs(a - b) / a < 0.02, (
        f"pv_score_ml no distingue medido de imputado: a={a:.4f} b={b:.4f}"
    )


def test_un_parseo_fallido_no_cuenta_como_convergido():
    """Regresión: se marcaba converged=True con gpw_parse_error registrado.

    Colaba en el recuento de la ronda y en la detección de runner atascado.
    """
    fuente = (ROOT / "src" / "buho" / "dft_jobs" / "collect_results.py").read_text(
        encoding="utf-8")
    bloque = fuente.split('row["error_message"] = f"gpw_parse_error')[1].split("def ")[0]
    assert 'row["converged"] = True' not in bloque, (
        "un job cuyo resumen no se pudo leer no dio la ciencia que se le pidió"
    )


def test_el_techo_de_sklearn_protege_al_usuario_no_solo_a_CI():  # noqa: N802
    """Los models/*.pkl no se despicklizan con scikit-learn 1.9.

    El tope estaba solo en el workflow de release: `pip install -e .` dejaba al
    usuario sin ningún modelo, en silencio.
    """
    import tomllib

    datos = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(datos["project"]["dependencies"])
    assert "scikit-learn>=1.8,<1.9" in deps.replace(" ", "")


def test_la_cascada_registra_por_log_no_solo_por_warnings():
    """En un servicio o GUI, un `warnings.warn` no llega a nadie."""
    fuente = (ROOT / "src" / "buho" / "screening" / "cascade.py").read_text(encoding="utf-8")
    bloque = fuente.split("def _load_surrogate")[1].split("def _runtime")[0]
    assert "log.warning" in bloque
