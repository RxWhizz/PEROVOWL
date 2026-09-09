"""Tests for the autonomous discovery loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _config(tmp_path: Path) -> Path:
    cfg = {
        "version": "test",
        "random_seed": 42,
        "chemical_space": {
            "A_sites": ["Cs", "MA"],
            "B_sites": ["Pb", "Sn"],
            "X_sites": ["I", "Br"],
        },
        "generation": {
            "fractions": [0.5],
            "fraction_mode": "continuous",
            "n_samples_per_combo": 1,
            "batch_size": 20,
            "modes": {
                "pure": True,
                "A_mixed": True,
                "B_mixed": True,
                "X_mixed": True,
                "multi_mixed": False,
            },
        },
        "filters": {
            "goldschmidt": {"min": 0.0, "max": 10.0},
            "octahedral": {"min": 0.0, "max": 10.0},
            "volume_A3": {"min": 0.0, "max": 10000.0},
        },
        "screening": {
            "tier1_surrogate": False,
            "tier2_mlff": False,
            "tier1_gate": False,
            "tier2_gate": False,
            "n_dft_per_batch": 0,
        },
        "acquisition": {"beta": 1.0, "pv_window": [1.1, 1.8]},
        "discovery": {
            "output_dir": "data/discovery",
            "dft_per_round": 3,
            "min_pv_score": 0.0,
            "frontier_size": 20,
            "pareto_input_size": 20,
            "mlff_pool_size": 0,
            "require_mlff_for_dft": False,
            "space": {
                "fraction_step": 0.5,
                "min_fraction": 0.5,
                "max_fraction": 0.5,
                "include_multi_mixed": False,
            },
        },
        "paths": {"runs_batches_dir": "runs/batches"},
    }
    path = tmp_path / "generator.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_chemical_space_enumerator_is_finite_and_reproducible(tmp_path):
    from buho.discovery.space import ChemicalSpaceEnumerator

    cfg = _config(tmp_path)
    enum1 = ChemicalSpaceEnumerator(cfg)
    enum2 = ChemicalSpaceEnumerator(cfg)
    c1, stats1 = enum1.enumerate()
    c2, stats2 = enum2.enumerate()

    assert stats1.total_generated == stats2.total_generated
    assert stats1.physically_viable == stats2.physically_viable
    assert stats1.physically_viable > 0
    assert [c.candidate_id for c in c1] == [c.candidate_id for c in c2]
    assert len({c.candidate_id for c in c1}) == len(c1)


def test_chemical_space_enumerator_accepts_in_memory_config(tmp_path):
    from buho.discovery.space import ChemicalSpaceEnumerator

    cfg_path = _config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["chemical_space"]["A_sites"] = ["Cs"]
    cfg["discovery"]["space"]["fraction_step"] = 0.5

    candidates, stats = ChemicalSpaceEnumerator(cfg).enumerate()

    assert stats.physically_viable == len(candidates)
    assert {tuple(c.A_site_species) for c in candidates} == {("Cs",)}


def test_discovery_init_score_and_dry_run_round(tmp_path):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)

    status = loop.init_space(reset=True)
    assert status["coverage"]["total"] > 0
    assert Path(status["paths"]["ledger"]).is_file()

    screen = loop.score_space(use_mlff=False)
    assert screen["n_frontier"] > 0

    chosen = loop.select_next_batch()
    assert 1 <= len(chosen) <= 3
    assert len(chosen) == len(set(chosen))

    prepared = loop.prepare_round(0, chosen, dry_run=True)
    assert prepared["n_selected"] == len(chosen)
    ledger = pd.read_csv(tmp_path / "data" / "discovery" / "ledger.csv")
    assert set(ledger[ledger["candidate_id"].isin(chosen)]["status"]) == {"dft_selected"}
    status_after = loop.status()
    assert status_after["queue"]
    assert set(item["status"] for item in status_after["queue"]) == {"dft_selected"}


def test_ledger_accepts_text_drop_reasons_after_csv_roundtrip(tmp_path):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)
    ledger = loop._read_ledger()
    cid = str(ledger.iloc[0]["candidate_id"])
    scored = pd.DataFrame([
        {
            "candidate_id": cid,
            "Eg_surrogate_eV": 2.13,
            "Eg_sigma_eV": 0.19,
            "Eform_eV_atom": 0.0,
            "Eform_std_eV_atom": 0.0,
            "meff_e_pred_m0": 0.3,
            "meff_h_pred_m0": 0.4,
            "eps_inf_pred": 12.0,
            "exciton_binding_meV": 40.0,
            "band_score": 0.1,
            "stab_score": 1.0,
            "transport_score": 0.9,
            "dielectric_score": 0.6,
            "exciton_score": 0.8,
            "pv_score_ml": 0.2,
            "acquisition_score": 0.2,
            "mlff_evaluated": False,
            "dropped_at_tier": "tier1",
            "drop_reason": "Eg fuera de ventana",
            "passed_eform": True,
        }
    ])

    loop._update_ledger_after_screen(ledger, scored, round_id=0)

    out = loop._read_ledger()
    row = out[out["candidate_id"] == cid].iloc[0]
    assert row["drop_reason"] == "Eg fuera de ventana"
    assert row["status"] == "screened"


def test_real_run_prepares_existing_dry_run_selection(tmp_path):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)
    loop.score_space(use_mlff=False)
    chosen = loop.select_next_batch()
    loop.prepare_round(0, chosen, dry_run=True)

    captured = {}

    def fake_prepare_round(round_id, candidate_ids, *, start_runner=True, dry_run=False):
        captured["round_id"] = round_id
        captured["candidate_ids"] = list(candidate_ids)
        captured["start_runner"] = start_runner
        captured["dry_run"] = dry_run
        state = loop._load_state()
        state["status"] = "dft_prepared"
        loop._save_state(state)
        return {"round_id": round_id, "n_selected": len(candidate_ids), "n_prepared": len(candidate_ids)}

    loop.prepare_round = fake_prepare_round  # type: ignore[method-assign]
    loop.advance(start_runner=False, dry_run=False, use_mlff=False)

    assert captured == {
        "round_id": 0,
        "candidate_ids": chosen,
        "start_runner": False,
        "dry_run": False,
    }


def test_discovery_marks_stale_runner_when_jobs_never_launch(tmp_path):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)
    state = loop._load_state()
    state.update(status="dft_running", active_round=0, current_round=0)
    loop._save_state(state)

    runs_dir = loop._round_runs_dir(0)
    runs_dir.mkdir(parents=True)
    for index in range(3):
        job = runs_dir / f"job{index}"
        job.mkdir()
        (job / "status.json").write_text(
            json.dumps({"status": "pending", "candidate_id": f"job{index}"}),
            encoding="utf-8",
        )
    (runs_dir / "runner.out").write_text(
        "No se encuentran los datasets PAW de GPAW.\nSe buscó Cs.PBE.gz en:\n",
        encoding="utf-8",
    )

    status = loop.status()

    assert status["runner"]["stale"] is True
    assert status["runner"]["status_counts"] == {"pending": 3}
    assert status["runner"]["error"] == "No se encuentran los datasets PAW de GPAW."

    recovered = loop.advance(start_runner=False, dry_run=False, use_mlff=False)
    assert recovered["state"]["status"] == "dft_prepared"
    assert "datasets PAW" in recovered["state"]["runner_error"]


def test_discovery_translates_configured_posix_mount_on_windows(tmp_path, monkeypatch):
    from buho.discovery import DiscoveryLoop
    from buho.discovery import engine as discovery_engine

    cfg_path = _config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["paths"]["runs_batches_dir"] = "/media/luis-ochoa/Nuevo vol/dft/runs/batches"
    cfg["discovery"]["windows_mounts"] = [
        {"posix": "/media/luis-ochoa/Nuevo vol", "windows": str(tmp_path / "NuevoVol")}
    ]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(discovery_engine.sys, "platform", "win32")

    loop = DiscoveryLoop(config_path=cfg_path, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)

    assert loop.dft_runs_dir == tmp_path / "NuevoVol" / "dft" / "runs" / "batches" / "discovery"


def test_discovery_builds_wsl_runner_command_with_mounts(tmp_path):
    from buho.discovery import DiscoveryLoop

    cfg_path = _config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["discovery"]["runner_backend"] = "wsl"
    cfg["discovery"]["runner_launcher"] = "direct"
    cfg["discovery"]["wsl"] = {
        "distro": "Ubuntu",
        "driver_python": "python3",
        "project_root": "/mnt/c/repo/PEROVOWL",
        "setup_path": "/mnt/n/gpaw-setups",
        "mounts": [
            {"windows": "C:/NuevoVol", "wsl": "/mnt/n"},
            {"windows": "C:/repo/PEROVOWL", "wsl": "/mnt/c/repo/PEROVOWL"},
        ],
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loop = DiscoveryLoop(config_path=cfg_path, project_root=Path("C:/repo/PEROVOWL"), data_root=tmp_path, models_root=tmp_path)

    assert loop._windows_path_to_wsl("C:/NuevoVol/dft/runs/batches") == "/mnt/n/dft/runs/batches"
    backend, cmd, _cwd = loop._runner_command(Path("C:/NuevoVol/dft/runs/batches/discovery/round_000"))
    shell_line = cmd[-1]

    assert backend == "wsl"
    assert cmd[:3] == ["wsl.exe", "-d", "Ubuntu"]
    assert "--relax-dir /mnt/n/dft/runs/batches/discovery/round_000" in shell_line
    assert "--setup-path /mnt/n/gpaw-setups" in shell_line


def test_launch_runner_preflight_failure_does_not_spawn(tmp_path, monkeypatch):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)
    runs_dir = loop._round_runs_dir(0)
    launched = []

    class FailedPreflight:
        returncode = 2
        stdout = "Preflight DFT fallo: No module named gpaw\n"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FailedPreflight())
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: launched.append(a) or None)

    with pytest.raises(RuntimeError, match="No module named gpaw"):
        loop._launch_runner(runs_dir)

    assert launched == []
    assert "Preflight DFT fallo" in (runs_dir / "runner.out").read_text(encoding="utf-8")
    assert (runs_dir / "runner_command.json").is_file()


def test_advance_persists_runner_error_when_prepared_round_fails(tmp_path):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)
    ledger = loop._read_ledger()
    cid = str(ledger.iloc[0]["candidate_id"])
    ledger.loc[ledger["candidate_id"].astype(str) == cid, "status"] = "dft_running"
    loop._write_ledger(ledger)
    state = loop._load_state()
    state.update(status="dft_prepared", active_round=0, current_round=0)
    loop._save_state(state)
    round_dir = loop._round_dir(0)
    round_dir.mkdir(parents=True)
    (round_dir / "manifest.json").write_text(
        json.dumps({"candidate_ids": [cid]}),
        encoding="utf-8",
    )
    runs_dir = loop._round_runs_dir(0)
    job = runs_dir / cid
    job.mkdir(parents=True)
    (job / "status.json").write_text(
        json.dumps({"status": "pending", "candidate_id": cid}),
        encoding="utf-8",
    )

    def fail_launch(_runs_dir):
        raise RuntimeError("preflight roto")

    loop._launch_runner = fail_launch  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="preflight roto"):
        loop.advance(start_runner=True, dry_run=False, use_mlff=False)

    recovered = loop.status()
    assert recovered["state"]["status"] == "dft_prepared"
    assert "preflight roto" in recovered["state"]["runner_error"]
    repaired = loop._read_ledger()
    row = repaired[repaired["candidate_id"].astype(str) == cid].iloc[0]
    assert row["status"] == "dft_prepared"


def test_discovery_runner_accepts_environment_runtime_overrides(tmp_path, monkeypatch):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)
    monkeypatch.setenv("BUHO_DFT_BACKEND", "local")
    monkeypatch.setenv("BUHO_DFT_LAUNCHER", "direct")
    monkeypatch.setenv("BUHO_GPAW_PYTHON", "/opt/gpaw/bin/python")
    monkeypatch.setenv("BUHO_MPI_LAUNCHER", "/opt/gpaw/bin/mpiexec")
    monkeypatch.setenv("BUHO_GPAW_SETUP_PATH", "/opt/gpaw/setups")

    backend, cmd, _ = loop._runner_command(tmp_path / "runs" / "round_000", preflight_only=True)

    assert backend == "local"
    assert "--launcher" in cmd
    assert cmd[cmd.index("--launcher") + 1] == "direct"
    assert cmd[cmd.index("--python") + 1] == "/opt/gpaw/bin/python"
    assert cmd[cmd.index("--mpirun") + 1] == "/opt/gpaw/bin/mpiexec"
    assert cmd[cmd.index("--setup-path") + 1] == "/opt/gpaw/setups"


def test_discovery_service_saves_space_override(tmp_path):
    from monitor_api import paths
    from monitor_api.services import discovery as service

    cfg_path = _config(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generator.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    paths.set_data_root(tmp_path)
    service.reset_background_for_tests()
    try:
        saved = service.save_config({
            "A_sites": ["Cs"],
            "B_sites": ["Pb", "Sn"],
            "X_sites": ["I"],
            "min_fraction": 0.5,
            "max_fraction": 0.5,
            "fraction_step": 0.5,
            "dft_per_round": 2,
            "modes": {"pure": True, "A_mixed": False, "B_mixed": True, "X_mixed": False, "multi_mixed": False},
        })
        multi = service.preview_config({
            "A_sites": ["Cs", "Rb"],
            "B_sites": ["Pb"],
            "X_sites": ["I", "Br"],
            "min_fraction": 0.5,
            "max_fraction": 0.5,
            "fraction_step": 0.5,
            "modes": {"pure": False, "A_mixed": False, "B_mixed": False, "X_mixed": False, "multi_mixed": True},
        })
    finally:
        paths.reset_data_root()
        service.reset_background_for_tests()

    override = tmp_path / "data" / "discovery" / "space_config.json"
    assert override.is_file()
    assert saved["A_sites"] == ["Cs"]
    assert saved["dft_per_round"] == 2
    assert saved["preview"]["physically_viable"] > 0
    assert multi["include_multi_mixed"] is True


def test_pareto_front_drops_dominated_rows():
    from buho.discovery.pareto import pareto_front

    df = pd.DataFrame([
        {"candidate_id": "a", "band": 0.9, "stab": 0.8, "acquisition_score": 0.9},
        {"candidate_id": "b", "band": 0.7, "stab": 0.6, "acquisition_score": 0.8},
        {"candidate_id": "c", "band": 0.6, "stab": 0.95, "acquisition_score": 0.7},
    ])

    out = pareto_front(df, ["band", "stab"])
    assert set(out["candidate_id"]) == {"a", "c"}


def test_discovery_api_routes_dispatch(monkeypatch):
    fastapi = pytest.importorskip("fastapi", reason="requiere el extra [web]")
    assert fastapi
    from fastapi.testclient import TestClient

    from monitor_api.main import create_app
    from monitor_api.services import discovery as service

    payload = {
        "state": {"status": "idle", "current_round": 0},
        "counts": {"unseen": 2},
        "coverage": {"total": 2, "seen": 0, "percent": 0.0},
        "frontier": [],
        "queue": [],
        "paths": {},
        "background": {"running": False, "last_error": None},
    }
    called = {}

    monkeypatch.setattr(service, "status", lambda: payload)
    monkeypatch.setattr(service, "init", lambda reset=False: {**payload, "reset": reset})
    monkeypatch.setattr(service, "start", lambda **kw: called.setdefault("start", kw) or payload)
    monkeypatch.setattr(service, "pause", lambda: {**payload, "state": {"status": "paused"}})
    monkeypatch.setattr(service, "resume", lambda: payload)
    monkeypatch.setattr(service, "frontier", lambda limit=100: {"items": [{"candidate_id": "a"}]})
    monkeypatch.setattr(service, "export", lambda: {"report": "r.md", "ledger": "l.csv", "frontier": "f.csv"})
    monkeypatch.setattr(
        service,
        "current_config",
        lambda: {"A_sites": ["Cs"], "B_sites": ["Pb"], "X_sites": ["I"]},
    )
    monkeypatch.setattr(
        service,
        "preview_config",
        lambda update=None: {"preview": {"physically_viable": 1}, "update": update or {}},
    )
    monkeypatch.setattr(
        service,
        "save_config",
        lambda update: {"override_path": "space_config.json", "saved": update},
    )

    app = create_app(config={})
    app.state.poller = None
    app.state.hub = None
    client = TestClient(app)

    assert client.get("/api/discovery/config").json()["A_sites"] == ["Cs"]
    preview = client.post("/api/discovery/config/preview", json={"A_sites": ["Cs"]}).json()
    assert preview["preview"]["physically_viable"] == 1
    saved = client.post("/api/discovery/config", json={"B_sites": ["Pb", "Sn"]}).json()
    assert saved["saved"]["B_sites"] == ["Pb", "Sn"]
    assert client.get("/api/discovery/status").json()["state"]["status"] == "idle"
    assert client.post("/api/discovery/init", json={"reset": True}).json()["reset"] is True
    assert client.post("/api/discovery/run", json={"dry_run": True, "start_runner": False}).status_code == 200
    assert called["start"]["dry_run"] is True
    assert client.post("/api/discovery/pause").json()["state"]["status"] == "paused"
    assert client.post("/api/discovery/resume").status_code == 200
    assert client.get("/api/discovery/frontier?limit=1").json()["items"][0]["candidate_id"] == "a"
    assert client.post("/api/discovery/export").json()["report"] == "r.md"


def test_discovery_report_export(tmp_path):
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path, models_root=tmp_path)
    loop.init_space(reset=True)
    loop.score_space(use_mlff=False)
    result = loop.export()

    report = Path(result["report"])
    assert report.is_file()
    assert "PEROVOWL Discovery Loop" in report.read_text(encoding="utf-8")


def test_la_fase_elegida_sobrevive_el_viaje_por_csv(tmp_path):
    """El ledger decia "CsPbI3: 1.5 eV" sin decir de cual de sus fases, que es
    media respuesta. Se anota al PREPARAR y no al recoger: la fase se decide
    ahi, y asi sobrevive a un DFT que falle --- que es justo cuando hace falta
    saber sobre que estructura fallo."""
    import json as _json

    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path,
                         models_root=tmp_path)
    loop.init_space(reset=True)
    cid = str(loop._read_ledger().iloc[0]["candidate_id"])

    job = tmp_path / "jobs" / cid
    job.mkdir(parents=True)
    (job / "fases.json").write_text(_json.dumps({
        "ok": True, "fase": "ortorrombica", "glazer": "a-a-c+",
        "grupo_espacial": {"disponible": True, "numero": 62, "simbolo": "Pnma"},
    }), encoding="utf-8")

    loop._anotar_fases([job])

    fila = loop._read_ledger().set_index("candidate_id").loc[cid]
    assert fila["fase"] == "ortorrombica"
    assert fila["grupo_espacial"] == "Pnma"


def test_un_job_sin_fase_no_inventa_una(tmp_path):
    """Sin potencial se prepara la cubica y no se escribe fases.json. El ledger
    tiene que decir que no se midio, no suponer que salio cubica."""
    from buho.discovery import DiscoveryLoop

    cfg = _config(tmp_path)
    loop = DiscoveryLoop(config_path=cfg, project_root=ROOT, data_root=tmp_path,
                         models_root=tmp_path)
    loop.init_space(reset=True)
    cid = str(loop._read_ledger().iloc[0]["candidate_id"])

    job = tmp_path / "jobs" / cid
    job.mkdir(parents=True)
    loop._anotar_fases([job])

    out = loop._read_ledger()
    assert "fase" not in out.columns or pd.isna(out.iloc[0].get("fase"))
