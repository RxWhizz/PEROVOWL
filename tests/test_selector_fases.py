"""El enganche de la competencia de fases al protocolo.

`buho.structure.fases` sabia elegir fase y estaba probado, pero nadie lo
llamaba: ningun sitio de produccion pasaba `selector_fase`, asi que el bucle
autonomo preparaba la cubica. Cerrar ese hueco obliga a cruzar una frontera
--- `seleccionar_fase` necesita un calculador ASE en proceso y el potencial vive
en el entorno del MLFF, en WSL--- y lo que se prueba aqui es el cruce: el
protocolo del worker, el lote unico, y que un entorno sin MLFF no cambie nada.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buho import mlff_runtime  # noqa: E402
from buho.mlff_runtime import MLFFRuntime, MLFFUnavailableError  # noqa: E402
from buho.structure import fases as F  # noqa: E402
from buho.structure import selector_mlff  # noqa: E402

CONFIG = ROOT / "config" / "generator.yaml"
WORKER = ROOT / "scripts" / "buho_mlff_worker.py"


# ── El protocolo del worker ──────────────────────────────────────────────────

def test_el_worker_acepta_fases_y_responde_una_sola_linea_json():
    """La respuesta viaja por stdout mezclada con lo que escriben matgl, dgl y
    el arranque de WSL. Con el lote vacio no hace falta el potencial, asi que
    esto se puede comprobar en cualquier maquina."""
    proc = subprocess.run(
        [sys.executable, str(WORKER), "--fases"],
        input=json.dumps({"estructuras": []}),
        capture_output=True, text=True, timeout=120,
    )

    lineas = [l for l in proc.stdout.splitlines() if l.strip()]
    assert len(lineas) == 1, f"stdout no es una sola linea: {proc.stdout!r}"
    payload = json.loads(lineas[0])
    assert payload == {"status": "ok", "resultados": []}
    assert proc.returncode == 0


def test_sin_orden_el_mensaje_nombra_las_tres():
    proc = subprocess.run([sys.executable, str(WORKER)],
                          capture_output=True, text=True, timeout=120)

    payload = json.loads(proc.stdout.strip())
    assert payload["status"] == "error"
    assert "--fases" in payload["error"]
    assert proc.returncode == 1


# ── El runtime ───────────────────────────────────────────────────────────────

def test_fases_usa_su_propio_timeout(monkeypatch):
    """Elegir fase no es predecir: cada estructura son cinco relajaciones FIRE
    sobre celdas de 40 atomos. Con el timeout de prediccion se cortaria a mitad
    y no habria forma de distinguirlo de un entorno roto."""
    visto = {}

    def _run(self, args, *, stdin=None, timeout=None):
        visto.update(args=args, stdin=stdin, timeout=timeout)
        return {"status": "ok", "resultados": [{"candidate_id": "x", "ok": True}]}

    # El dataclass es frozen: se parchea en la clase, no en la instancia.
    monkeypatch.setattr(MLFFRuntime, "_run", _run)
    rt = MLFFRuntime(backend="local", python="p", worker="w", timeout=900,
                     timeout_fases=3600)

    out = rt.fases([{"candidate_id": "x"}], opciones={"fmax": 0.05})

    assert out == [{"candidate_id": "x", "ok": True}]
    assert visto["args"] == ["--fases"]
    assert visto["timeout"] == 3600
    assert json.loads(visto["stdin"])["opciones"] == {"fmax": 0.05}


def test_un_lote_vacio_no_lanza_proceso(monkeypatch):
    def _explota(self, *a, **k):
        raise AssertionError("no deberia ejecutarse nada")

    monkeypatch.setattr(MLFFRuntime, "_run", _explota)
    assert MLFFRuntime(backend="local", python="p", worker="w").fases([]) == []


def test_el_timeout_de_fases_sale_de_la_config():
    rt = mlff_runtime.resolve(
        {"discovery": {"mlff": {"backend": "off", "timeout_fases": 120}}})
    assert rt.timeout_fases == 120


# ── El selector ──────────────────────────────────────────────────────────────

class _RuntimeFalso:
    """Un MLFF que responde sin WSL ni potencial, y cuenta las llamadas."""

    def __init__(self, respuesta, *, revienta=False):
        self.backend = "local"
        self.llamadas: list[list[dict]] = []
        self._respuesta = respuesta
        self._revienta = revienta

    def fases(self, estructuras, *, opciones=None, timeout=None):
        self.llamadas.append(estructuras)
        if self._revienta:
            raise MLFFUnavailableError("no hay entorno MLFF")
        return self._respuesta(estructuras)


@pytest.fixture(scope="module")
def cfg():
    import yaml
    if not CONFIG.is_file():
        pytest.skip("config/generator.yaml no disponible")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def candidatos(cfg):
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    hechos = []
    for x in ("I", "Br", "Cl"):
        hechos.append(gen._make_candidate(
            A_sp=["Cs"], B_sp=["Pb"], X_sp=[x],
            A_f={"Cs": 1.0}, B_f={"Pb": 1.0}, X_f={x: 1.0}, mode="pure"))
    return hechos


def _ganadora_ortorrombica(cfg, candidato):
    """La geometria que devolveria el worker, generada de verdad."""
    from buho.structure.build_abx3 import ABX3StructureBuilder

    atoms, meta = ABX3StructureBuilder(cfg).build(candidato, out_dir=None, export=False)
    todas = F.generar_candidatas(atoms, b_sites={"Pb"}, x_sites={"I", "Br", "Cl"})
    orto = next(c for c in todas if c["fase"] == "ortorrombica")["atoms"]
    return {
        "symbols": list(orto.get_chemical_symbols()),
        "positions": [[float(v) for v in f] for f in orto.get_positions()],
        "cell": [[float(v) for v in f] for f in orto.get_cell()],
        "pbc": [True, True, True],
    }


def test_n_candidatos_una_sola_llamada(cfg, candidatos):
    """Es la razon de ser del lote: cargar el potencial cuesta mucho mas que
    usarlo, y en WSL hay que sumar el arranque de la distro."""
    rt = _RuntimeFalso(lambda es: [
        {"candidate_id": e["candidate_id"], "ok": True, "fase": "cubica",
         "geometria": {"symbols": e["symbols"], "positions": e["positions"],
                       "cell": e["cell"], "pbc": e["pbc"]}}
        for e in es])
    sel = selector_mlff.crear_selector_fase(cfg, candidatos, runtime=rt)

    for c in candidatos:
        assert sel(c, None, {})["ok"]

    assert len(rt.llamadas) == 1
    assert len(rt.llamadas[0]) == len(candidatos)


def test_el_lote_lleva_la_semilla_y_los_sitios(cfg, candidatos):
    """El potencial aporta la forma; el tamano lo pone la semilla calibrada. Si
    no viajara, la celda que va a DFT seria la que infla el potencial."""
    rt = _RuntimeFalso(lambda es: [])
    sel = selector_mlff.crear_selector_fase(cfg, candidatos, runtime=rt)
    sel(candidatos[0], None, {})

    enviado = rt.llamadas[0][0]
    assert enviado["b_sites"] == ["Pb"]
    assert enviado["x_sites"] == ["I"]
    assert 5.0 < enviado["a_semilla_A"] < 7.0
    assert len(enviado["symbols"]) == 5, "el padre es la celda primitiva"


def test_el_grupo_espacial_se_reidentifica_en_el_motor(cfg, candidatos):
    """spglib viaja en el binario del motor pero no tiene por que estar en el
    entorno del MLFF, donde `grupo_espacial` degrada a 'no disponible'."""
    pytest.importorskip("spglib")
    geometria = _ganadora_ortorrombica(cfg, candidatos[0])
    rt = _RuntimeFalso(lambda es: [{
        "candidate_id": es[0]["candidate_id"], "ok": True,
        "fase": "ortorrombica", "glazer": "a-a-c+",
        "grupo_espacial": {"disponible": False, "motivo": "sin spglib"},
        "geometria": geometria,
    }])
    sel = selector_mlff.crear_selector_fase(cfg, [candidatos[0]], runtime=rt)

    r = sel(candidatos[0], None, {})

    assert r["grupo_espacial"]["numero"] == 62
    assert len(r["atoms"]) == 40


def test_un_fallo_del_lote_no_se_reintenta_por_candidato(cfg, candidatos, caplog):
    """Sin cachear el fallo, cada candidato volveria a arrancar WSL para recibir
    el mismo error N veces."""
    rt = _RuntimeFalso(lambda es: [], revienta=True)
    sel = selector_mlff.crear_selector_fase(cfg, candidatos, runtime=rt)

    with caplog.at_level("WARNING"):
        salidas = [sel(c, None, {}) for c in candidatos]

    assert len(rt.llamadas) == 1
    assert all(not s["ok"] for s in salidas)
    assert "se preparan las cubicas" in caplog.text


def test_el_interruptor_de_entorno_evita_salir_a_buscar(cfg, candidatos, monkeypatch):
    """Devolver None es el camino documentado: el preparador usa la cubica.

    El interruptor apaga solo la BUSQUEDA de un MLFF en esta maquina, que es lo
    unico que puede arrancar WSL y tardar minutos. Con un runtime inyectado no
    hay nada que buscar y la seleccion sigue funcionando --- si no, esta misma
    suite no podria probarla.
    """
    monkeypatch.setenv("BUHO_FASES", "off")

    assert selector_mlff.crear_selector_fase(cfg, candidatos) is None

    rt = _RuntimeFalso(lambda es: [])
    assert selector_mlff.crear_selector_fase(cfg, candidatos, runtime=rt) is not None


def test_un_backend_apagado_no_da_selector(cfg, candidatos):
    class _Apagado:
        backend = "off"

    assert selector_mlff.crear_selector_fase(cfg, candidatos, runtime=_Apagado()) is None


def test_desactivarlo_en_la_config_tambien(cfg, candidatos):
    apagado = {**cfg, "discovery": {**cfg.get("discovery", {}), "fases": {"activo": False}}}
    rt = _RuntimeFalso(lambda es: [])

    assert selector_mlff.crear_selector_fase(apagado, candidatos, runtime=rt) is None


def test_un_candidato_que_el_worker_no_devuelve_cae_a_la_cubica(cfg, candidatos):
    rt = _RuntimeFalso(lambda es: [])
    sel = selector_mlff.crear_selector_fase(cfg, candidatos, runtime=rt)

    r = sel(candidatos[0], None, {})

    assert r["ok"] is False
    assert "no devolvio" in r["motivo"]
