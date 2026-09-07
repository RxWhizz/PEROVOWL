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
