"""Fases candidatas: que el pipeline sepa en cuál está, en vez de suponerla.

El pipeline construía siempre Pm-3m. El propio `riesgo_politipo` ya avisaba de
que eso no basta, con el contraejemplo de casa: CsPbI₃ tiene t = 0.851, dentro de
rango, y su fase estable a 25 °C es la δ ortorrómbica, no la α cúbica que se
evaluaba.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buho.structure import fases as F  # noqa: E402

CONFIG = ROOT / "config" / "generator.yaml"

#: Grupo espacial que debe salir de cada sistema de inclinación. Es la
#: comprobación de que la siembra reproduce la cristalografía, no una etiqueta.
GRUPOS = {
    "cubica": (221, "Pm-3m"),
    "tetragonal_antifase": (140, "I4/mcm"),
    "tetragonal_en_fase": (127, "P4/mbm"),
    "romboedrica": (167, "R-3c"),
    "ortorrombica": (62, "Pnma"),
}


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def padre_cubico(cfg):
    """CsPbI₃ cúbico, el punto de partida de todas las fases."""
    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    cand = gen._make_candidate(
        A_sp=["Cs"], B_sp=["Pb"], X_sp=["I"],
        A_f={"Cs": 1.0}, B_f={"Pb": 1.0}, X_f={"I": 1.0}, mode="pure",
    )
    atoms, _ = ABX3StructureBuilder(cfg).build(cand)
    return atoms


@pytest.fixture(scope="module")
def candidatas(padre_cubico):
    return {d["fase"]: d for d in F.generar_candidatas(
        padre_cubico, b_sites={"Pb"}, x_sites={"I"})}


def _spglib_o_skip():
    pytest.importorskip("spglib", reason="la identificación de simetría lo necesita")


# ── Lo que decide la fase ────────────────────────────────────────────────────

def test_el_padre_es_cubico(padre_cubico):
    _spglib_o_skip()
    assert F.grupo_espacial(padre_cubico)["numero"] == 221


@pytest.mark.parametrize("nombre", sorted(GRUPOS))
def test_cada_inclinacion_da_su_grupo_espacial(candidatas, nombre):
    """La siembra reproduce la cristalografía desde la notación de Glazer.

    Costó tres errores encadenados llegar aquí, y cada uno bajaba la simetría de
    forma distinta, así que conviene que quede fijado:

      1. Girar cada anión alrededor del B "más cercano" — pero un anión puente
         tiene dos B a la misma distancia, así que la elección era arbitraria.
      2. Encadenar tres rotaciones sobre x, y, z — no conmutan, y la composición
         no es la inclinación pretendida (a⁻a⁻a⁻ es UN giro sobre [111]).
      3. Promediar posiciones absolutas de aniones compartidos — dos octaedros
         en lados opuestos de la celda los ven como imágenes periódicas
         distintas, y la media caía en el centro de la celda.
    """
    _spglib_o_skip()
    numero, simbolo = GRUPOS[nombre]
    medido = F.grupo_espacial(candidatas[nombre]["atoms"])

    assert medido["disponible"] is True
    assert medido["numero"] == numero, (
        f"{nombre}: se esperaba {simbolo} ({numero}), salió "
        f"{medido['simbolo']} ({medido['numero']})")


def test_las_cinco_fases_son_simetrias_distintas(candidatas):
    """Sembrar dos veces la misma cuenca sería gastar una relajación de más."""
    _spglib_o_skip()
    numeros = {F.grupo_espacial(d["atoms"])["numero"] for d in candidatas.values()}
    assert len(numeros) == len(candidatas)


# ── Mecánica de la siembra ───────────────────────────────────────────────────

def test_la_composicion_no_cambia_al_inclinar(padre_cubico, candidatas):
    """Una inclinación mueve átomos; no crea ni destruye ninguno."""
    from collections import Counter

    proporcion = Counter(padre_cubico.get_chemical_symbols())
    n_padre = len(padre_cubico)
    for nombre, d in candidatas.items():
        c = Counter(d["atoms"].get_chemical_symbols())
        factor = len(d["atoms"]) / n_padre
        for especie, n in proporcion.items():
            assert c[especie] == pytest.approx(n * factor), f"{nombre}: {especie}"


def test_la_cubica_no_se_toca(padre_cubico, candidatas):
    """a⁰a⁰a⁰ es 'sin inclinar': tiene que salir idéntica al padre."""
    import numpy as np

    cubica = candidatas["cubica"]["atoms"]
    assert np.allclose(cubica.get_positions(), padre_cubico.get_positions())


def test_los_aniones_se_mueven_y_los_cationes_no(candidatas):
    """La inclinación gira los octaedros: mueve X, deja B en su sitio."""
    import numpy as np

    for nombre in ("tetragonal_antifase", "romboedrica"):
        atoms = candidatas[nombre]["atoms"]
        simbolos = np.array(atoms.get_chemical_symbols())
        # Los B siguen en posiciones de alta simetría (múltiplos de 1/(2n)).
        frac_b = atoms.get_scaled_positions()[simbolos == "Pb"]
        n = candidatas[nombre]["supercelda"][0]
        resto = np.abs(frac_b * 2 * n - np.round(frac_b * 2 * n))
        assert np.all(resto < 1e-6), f"{nombre}: los cationes B se movieron"


def test_sin_sitios_reconocibles_no_revienta(padre_cubico, caplog):
    """Una composición cuyos símbolos no coincidan no puede tumbar la ronda."""
    fase = F.FASES_INCLINACION[1]
    with caplog.at_level("WARNING"):
        salida = F.aplicar_inclinacion(
            padre_cubico, fase, b_sites={"Xx"}, x_sites={"Yy"})
    assert len(salida) == len(padre_cubico)
    assert "sin inclinar" in caplog.text


# ── La identificación ────────────────────────────────────────────────────────

def test_sin_spglib_se_dice_en_vez_de_fallar(padre_cubico, monkeypatch):
    """Sin la herramienta no se puede identificar la fase; hay que decirlo."""
    import builtins

    real = builtins.__import__

    def _sin_spglib(nombre, *a, **k):
        if nombre == "spglib":
            raise ImportError("no está")
        return real(nombre, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _sin_spglib)
    r = F.grupo_espacial(padre_cubico)
    assert r["disponible"] is False
    assert "spglib" in r["motivo"]


def test_la_tolerancia_se_reporta(padre_cubico):
    """Un grupo espacial sin la tolerancia con la que se midió no se puede
    interpretar: con symprec exacto casi todo lo relajado sale P1."""
    _spglib_o_skip()
    r = F.grupo_espacial(padre_cubico, tolerancia=0.01)
    assert r["tolerancia"] == 0.01


# ── Elegir la fase: el potencial la forma, la semilla el tamaño ──────────────

class _CalculadorFalso:
    """Prefiere la fase que se le diga, sin depender del MLFF real.

    Las pruebas con M3GNet costarían minutos y una descarga; lo que hay que
    fijar aquí es la lógica de selección, no el potencial.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, favorita: str, energias: dict[str, float]):
        self.favorita = favorita
        self.energias = energias
        self.atoms = None

    def get_potential_energy(self, atoms=None, **_):
        import numpy as np
        a = atoms if atoms is not None else self.atoms
        # La energía depende del número de átomos para que salga por fórmula.
        n_fu = max(1, len(a) // 5)
        # Se identifica la fase por cuánto se desvían los aniones de la posición
        # ideal: la cúbica no tiene desviación, las inclinadas sí.
        frac = a.get_scaled_positions()
        desvio = float(np.abs(frac * 4 - np.round(frac * 4)).max())
        clave = "cubica" if desvio < 1e-6 else "inclinada"
        return self.energias[clave] * n_fu

    def get_forces(self, atoms=None, **_):
        import numpy as np
        a = atoms if atoms is not None else self.atoms
        return np.zeros((len(a), 3))

    def get_stress(self, atoms=None, **_):
        import numpy as np
        return np.zeros(6)

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name in ("energy", "free_energy"):
            return self.get_potential_energy(atoms)
        if name == "forces":
            return self.get_forces(atoms)
        if name == "stress":
            return self.get_stress(atoms)
        raise NotImplementedError(name)

    def calculate(self, atoms=None, *a, **k):
        self.atoms = atoms

    def check_state(self, atoms, tol=1e-15):
        return []


def test_reescalar_conserva_el_grupo_espacial(candidatas):
    """El escalado uniforme mueve todos los átomos igual en fraccionarias.

    Es lo que permite quedarse con la forma que dio el potencial y el tamaño que
    dice la semilla, sin cambiar la simetría por el camino.
    """
    _spglib_o_skip()
    original = candidatas["ortorrombica"]["atoms"]
    antes = F.grupo_espacial(original)

    escalada = F.reescalar_a_volumen(original, original.get_volume() * 1.3)

    assert F.grupo_espacial(escalada)["numero"] == antes["numero"]
    assert escalada.get_volume() == pytest.approx(original.get_volume() * 1.3)


def test_reescalar_no_revienta_con_volumen_absurdo(candidatas):
    original = candidatas["cubica"]["atoms"]
    assert len(F.reescalar_a_volumen(original, 0.0)) == len(original)
    assert len(F.reescalar_a_volumen(original, -5.0)) == len(original)


def test_la_fase_elegida_lleva_el_tamano_de_la_semilla(padre_cubico):
    """El hallazgo de CsPbI₃: el potencial ordena bien pero infla la celda.

    M3GNet dejaba la cúbica en 6.4619 Å frente a 6.18 experimental (+4.56 %),
    cuando la semilla calibrada fallaba un +0.06 %. Como el gap es muy sensible
    al volumen, quedarse con esa celda desharía lo que gana el funcional. El
    potencial aporta la forma; la semilla, el tamaño.
    """
    _spglib_o_skip()
    a_semilla = 6.1834
    calc = _CalculadorFalso("inclinada", {"cubica": -10.0, "inclinada": -10.5})

    r = F.seleccionar_fase(padre_cubico, calc, b_sites={"Pb"}, x_sites={"I"},
                           a_semilla=a_semilla)

    assert r["ok"] is True
    n_fu = len(r["atoms"]) // 5
    a_final = (r["atoms"].get_volume() / n_fu) ** (1 / 3)
    assert a_final == pytest.approx(a_semilla, abs=1e-6), (
        "el tamaño tiene que venir de la semilla, no del potencial")


def test_gana_la_de_menor_energia(padre_cubico):
    _spglib_o_skip()
    calc = _CalculadorFalso("inclinada", {"cubica": -10.0, "inclinada": -10.5})
    r = F.seleccionar_fase(padre_cubico, calc, b_sites={"Pb"}, x_sites={"I"},
                           a_semilla=6.1834)
    assert r["fase"] != "cubica"
    assert r["ranking"][0]["dE_meV_por_formula"] == 0.0


def test_si_gana_la_cubica_se_respeta(padre_cubico):
    """No se fuerza una distorsión: si el potencial dice cúbica, cúbica."""
    _spglib_o_skip()
    calc = _CalculadorFalso("cubica", {"cubica": -11.0, "inclinada": -10.0})
    r = F.seleccionar_fase(padre_cubico, calc, b_sites={"Pb"}, x_sites={"I"},
                           a_semilla=6.1834)
    assert r["fase"] == "cubica"


def test_el_ranking_completo_queda_registrado(padre_cubico):
    """Saber que la segunda estaba a 3 meV o a 300 cambia cuánto fiarse."""
    _spglib_o_skip()
    calc = _CalculadorFalso("inclinada", {"cubica": -10.0, "inclinada": -10.5})
    r = F.seleccionar_fase(padre_cubico, calc, b_sites={"Pb"}, x_sites={"I"},
                           a_semilla=6.1834)
    assert len(r["ranking"]) == len(F.FASES_INCLINACION)
    assert all("dE_meV_por_formula" in x for x in r["ranking"])
    assert "atoms" not in r["ranking"][0], "el ranking es para registrar, no pesa estructuras"


def test_una_fase_que_falla_no_tumba_la_seleccion(padre_cubico, caplog):
    """Si una relajación revienta, las demás siguen compitiendo."""
    _spglib_o_skip()

    class _Rompe(_CalculadorFalso):
        def get_potential_energy(self, atoms=None, **k):
            a = atoms if atoms is not None else self.atoms
            if len(a) > 5:
                raise RuntimeError("boom")
            return -10.0

    with caplog.at_level("WARNING"):
        r = F.seleccionar_fase(padre_cubico, _Rompe("cubica", {}),
                               b_sites={"Pb"}, x_sites={"I"}, a_semilla=6.1834)
    assert r["ok"] is True
    assert r["fase"] == "cubica"
    assert "se descarta" in caplog.text


# ── El enganche a la preparación de trabajos DFT ─────────────────────────────

def _preparador(tmp_path, cfg, selector=None):
    from buho.dft_jobs.prepare_relaxation_jobs import RelaxationJobPreparer

    return RelaxationJobPreparer(cfg, project_root=tmp_path, selector_fase=selector)


def _candidato(gen):
    return gen._make_candidate(
        A_sp=["Cs"], B_sp=["Pb"], X_sp=["I"],
        A_f={"Cs": 1.0}, B_f={"Pb": 1.0}, X_f={"I": 1.0}, mode="pure")


def test_sin_selector_se_prepara_la_cubica_de_siempre(tmp_path, cfg):
    """Elegir fase necesita un potencial; sin él, el comportamiento no cambia."""
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    prep = _preparador(tmp_path, cfg)
    hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    assert len(hechos) == 1
    assert not (hechos[0] / "fases.json").exists()


def test_con_selector_va_a_dft_la_fase_elegida(tmp_path, cfg, candidatas):
    """El pipeline construía siempre Pm-3m. Para CsPbI₃ esa es la fase α, que
    solo existe por encima de 330 °C: a temperatura ambiente el material está en
    otra, con otro bandgap."""
    import json

    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    elegida = candidatas["ortorrombica"]["atoms"]

    def selector(cand, atoms, meta):
        return {"ok": True, "fase": "ortorrombica", "glazer": "a-a-c+",
                "grupo_espacial": {"simbolo": "Pnma", "numero": 62},
                "convergido": False, "a_semilla_A": 6.1834,
                "a_relajado_mlff_A": 6.3210, "atoms": elegida,
                "ranking": [{"fase": "ortorrombica", "dE_meV_por_formula": 0.0}]}

    prep = _preparador(tmp_path, cfg, selector)
    hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    guardado = json.loads((hechos[0] / "fases.json").read_text(encoding="utf-8"))
    assert guardado["fase"] == "ortorrombica"
    assert guardado["grupo_espacial"]["numero"] == 62
    assert "atoms" not in guardado, "el JSON registra, no serializa estructuras"


def test_un_selector_que_revienta_no_tumba_la_ronda(tmp_path, cfg, caplog):
    """Sin fase elegida se sigue con la cúbica, que es lo que había antes."""
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)

    def selector(cand, atoms, meta):
        raise RuntimeError("el potencial no cargó")

    prep = _preparador(tmp_path, cfg, selector)
    with caplog.at_level("WARNING"):
        hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    assert len(hechos) == 1
    assert "se usa la cubica" in caplog.text


def test_un_selector_que_no_elige_nada_tampoco(tmp_path, cfg, caplog):
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    prep = _preparador(
        tmp_path, cfg,
        lambda c, a, m: {"ok": False, "motivo": "ninguna fase relajo"})

    with caplog.at_level("WARNING"):
        hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    assert len(hechos) == 1
    assert "ninguna fase relajo" in caplog.text


# ── Que la fase elegida llegue de verdad al calculo ──────────────────────────
#
# Estas tres cubren el hueco que hacia cosmetica toda la seleccion: `prepare`
# exporta la cubica ANTES de elegir fase, el `input.py` generado hace
# `read("structure.cif")`, y `_elegir_fase` no reexportaba. Con un selector
# perfecto, GPAW seguia relajando la cubica y el unico rastro era un fases.json
# diciendo otra cosa. Las pruebas de arriba pasaban igual.

def _selector_a(atoms_elegidos, fase="ortorrombica"):
    def selector(cand, atoms, meta):
        return {"ok": True, "fase": fase, "glazer": "a-a-c+",
                "grupo_espacial": {"simbolo": "Pnma", "numero": 62},
                "convergido": True, "a_semilla_A": 6.1834,
                "a_relajado_mlff_A": 6.3210, "atoms": atoms_elegidos,
                "ranking": [{"fase": fase, "dE_meV_por_formula": 0.0}]}
    return selector


def test_el_cif_del_job_es_la_fase_elegida_no_la_cubica(tmp_path, cfg, candidatas):
    """Lo que GPAW va a leer. Es la prueba que faltaba: sin ella, elegir fase
    escribia un JSON y no cambiaba ni un atomo del calculo."""
    from ase.io import read

    from buho.generator.heuristic_generator import HeuristicGenerator

    _spglib_o_skip()
    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    elegida = candidatas["ortorrombica"]["atoms"]

    prep = _preparador(tmp_path, cfg, _selector_a(elegida))
    hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    del_disco = read(str(hechos[0] / "structure.cif"))
    assert len(del_disco) == len(elegida)
    assert F.grupo_espacial(del_disco)["numero"] == 62, (
        "el CIF que se va a calcular sigue siendo cubico")


def test_la_cubica_de_partida_se_conserva_para_poder_auditarla(tmp_path, cfg, candidatas):
    """De donde salio la celda es discutible; que no quede rastro, no."""
    from ase.io import read

    from buho.generator.heuristic_generator import HeuristicGenerator

    _spglib_o_skip()
    gen = HeuristicGenerator(str(CONFIG), random_seed=1)

    prep = _preparador(tmp_path, cfg, _selector_a(candidatas["ortorrombica"]["atoms"]))
    hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    padre = hechos[0] / "structure_cubica.cif"
    assert padre.is_file()
    assert F.grupo_espacial(read(str(padre)))["numero"] == 221


def test_una_fase_inclinada_se_calcula_como_supercelda(tmp_path, cfg, candidatas):
    """La bandera decide la malla k y el reparto MPI. Una fase inclinada es una
    supercelda 2x2x2 aunque el compuesto sea puro: muestrearla a [2,2,2] es una
    malla efectiva 4^3 --- ocho veces el coste, y distinta de la 2^3 a la que
    esta calibrada la escala de bandgap."""
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    elegida = candidatas["ortorrombica"]["atoms"]
    assert len(elegida) == 40, "la fixture debe ser una 2x2x2"

    prep = _preparador(tmp_path, cfg, _selector_a(elegida))
    hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    generado = (hechos[0] / "input.py").read_text(encoding="utf-8")
    assert "_kpts = [1, 1, 1] if True else [2, 2, 2]" in generado


def test_sin_seleccion_una_celda_pura_sigue_siendo_primitiva(tmp_path, cfg):
    """La otra mitad del contrato: el cambio no puede convertir en supercelda lo
    que siempre fue una celda de cinco atomos."""
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    prep = _preparador(tmp_path, cfg)
    hechos = prep.prepare([_candidato(gen)], out_root=tmp_path / "jobs")

    generado = (hechos[0] / "input.py").read_text(encoding="utf-8")
    assert "_kpts = [1, 1, 1] if False else [2, 2, 2]" in generado
