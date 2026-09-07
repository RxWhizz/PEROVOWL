"""Correspondencia entre la composición que se pide y la que se construye.

Todo esto sale de un hallazgo concreto: en el CSV de entrenamiento del bucle de
descubrimiento, 56 de 111 filas compartían Eg = 1.075 eV con 48 fórmulas
distintas. La causa era que una supercelda 2×2×2 tiene 8 sitios A, así que
`round(0.051 * 8) = 0`: `Cs0.95Rb0.051SnI3` se construía como CsSnI3 puro y su
DFT se archivaba bajo la fórmula del dopado.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buho.structure.occupancy import (  # noqa: E402
    ajustar_fraccion,
    clave_estructura,
    cuantizar_ocupacion,
    especies_perdidas,
    fracciones_realizadas,
    sitios_por_subred,
)

CONFIG = ROOT / "config" / "generator.yaml"


@pytest.fixture(scope="module")
def cfg():
    with CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── La cuantización en sí ────────────────────────────────────────────────────

def test_sitios_por_subred_cuenta_la_supercelda():
    assert sitios_por_subred([1, 1, 1]) == {"A": 1, "B": 1, "X": 3}
    assert sitios_por_subred([2, 2, 2]) == {"A": 8, "B": 8, "X": 24}
    assert sitios_por_subred([3, 3, 3]) == {"A": 27, "B": 27, "X": 81}


def test_un_dopante_por_debajo_de_la_resolucion_se_queda_sin_atomos():
    """El hecho que originó todo: hay que poder detectarlo, no que no ocurra."""
    cuentas = dict(cuantizar_ocupacion(["Cs", "Rb"], {"Cs": 0.949, "Rb": 0.051}, 8))
    assert cuentas == {"Cs": 8, "Rb": 0}
    assert especies_perdidas(["Cs", "Rb"], {"Cs": 0.949, "Rb": 0.051}, 8) == ["Rb"]


def test_la_ocupacion_siempre_suma_los_sitios_disponibles():
    for fr in ({"I": 0.5, "Br": 0.3, "Cl": 0.2}, {"I": 0.99, "Br": 0.005, "Cl": 0.005}):
        cuentas = cuantizar_ocupacion(["I", "Br", "Cl"], fr, 24)
        assert sum(n for _, n in cuentas) == 24


def test_fracciones_realizadas_dicen_lo_que_hay_no_lo_que_se_pidio():
    assert fracciones_realizadas(["Cs", "Rb"], {"Cs": 0.949, "Rb": 0.051}, 8) == {
        "Cs": 1.0,
        "Rb": 0.0,
    }


# ── El ajuste que evita el problema ──────────────────────────────────────────

def test_ajustar_fraccion_nunca_vacia_una_especie_pedida():
    """Devolver 0 convertiría la mezcla en un endmember mal etiquetado."""
    for f in (0.001, 0.02, 0.051, 0.0624):
        ajustada = ajustar_fraccion(f, 8)
        assert ajustada >= 1 / 8
        assert ajustada <= 7 / 8
    for f in (0.999, 0.98, 0.95):
        assert ajustar_fraccion(f, 8) <= 7 / 8


@pytest.mark.parametrize("n_sitios", [8, 24, 27])
def test_lo_ajustado_se_construye_exactamente(n_sitios):
    """Ida y vuelta: pedir una fracción ajustada da esa misma fracción."""
    for k in range(1, n_sitios):
        f = ajustar_fraccion(k / n_sitios, n_sitios)
        realizadas = fracciones_realizadas(["S1", "S2"], {"S1": f, "S2": 1.0 - f}, n_sitios)
        assert realizadas["S1"] == pytest.approx(f, abs=1e-9)
        assert realizadas["S2"] > 0.0


# ── Identidad estructural ────────────────────────────────────────────────────

def _candidato(gen, A_sp, A_f, modo="A_mixed"):
    return gen._make_candidate(
        A_sp=A_sp, B_sp=["Sn"], X_sp=["I"],
        A_f=A_f, B_f={"Sn": 1.0}, X_f={"I": 1.0}, mode=modo,
    )


def test_dos_formulas_distintas_con_la_misma_estructura_comparten_clave():
    """La fórmula no sirve como identidad: por eso se gastaba DFT dos veces."""
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    dopado = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.949, "Rb": 0.051})
    puro = _candidato(gen, ["Cs", "Rb"], {"Cs": 1.0, "Rb": 0.0})

    assert dopado.formula != puro.formula
    assert clave_estructura(dopado, [2, 2, 2]) == clave_estructura(puro, [2, 2, 2])


def test_una_diferencia_representable_si_cambia_la_clave():
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    uno = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.875, "Rb": 0.125})
    otro = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.75, "Rb": 0.25})

    assert clave_estructura(uno, [2, 2, 2]) != clave_estructura(otro, [2, 2, 2])


# ── El constructor ───────────────────────────────────────────────────────────

def test_el_constructor_avisa_y_deja_constancia_de_lo_que_perdio(cfg, caplog):
    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    cand = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.949, "Rb": 0.051})

    with caplog.at_level("WARNING"):
        atoms, meta = ABX3StructureBuilder(cfg).build(cand)

    assert "Rb" in caplog.text, "una especie que desaparece no puede pasar en silencio"
    assert Counter(atoms.get_chemical_symbols())["Rb"] == 0
    assert meta["species_dropped"] == ["Rb"]
    assert meta["composition_exact"] is False
    assert meta["fractions_realized"]["A"] == {"Cs": 1.0, "Rb": 0.0}


def test_una_composicion_representable_se_construye_exacta(cfg):
    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    cand = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.75, "Rb": 0.25})
    atoms, meta = ABX3StructureBuilder(cfg).build(cand)

    assert Counter(atoms.get_chemical_symbols())["Rb"] == 2
    assert meta["composition_exact"] is True
    assert meta["species_dropped"] == []


def test_la_celda_se_dimensiona_con_lo_que_contiene(cfg):
    """Un dopante que no está no puede seguir estirando la red."""
    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    builder = ABX3StructureBuilder(cfg)
    # El sitio B sí entra en el parámetro de red (r_B). Pedir un 5 % de Ge que
    # no cabe en 8 sitios no puede encoger la celda de CsSnI3.
    con_dopante = gen._make_candidate(
        A_sp=["Cs"], B_sp=["Sn", "Ge"], X_sp=["I"],
        A_f={"Cs": 1.0}, B_f={"Sn": 0.95, "Ge": 0.05}, X_f={"I": 1.0}, mode="B_mixed",
    )
    _, meta_dop = builder.build(con_dopante)
    puro = gen._make_candidate(
        A_sp=["Cs"], B_sp=["Sn"], X_sp=["I"],
        A_f={"Cs": 1.0}, B_f={"Sn": 1.0}, X_f={"I": 1.0}, mode="pure",
    )
    _, meta_puro = builder.build(puro)

    assert meta_dop["species_dropped"] == ["Ge"]
    assert meta_dop["lattice_constant_A"] == pytest.approx(
        meta_puro["lattice_constant_A"], abs=1e-6
    )


# ── El generador ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sitio", ["A", "B", "X"])
def test_el_generador_solo_propone_lo_que_la_supercelda_puede_contener(sitio):
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=3)
    n = sitios_por_subred([2, 2, 2])[sitio]
    propuestas = gen._binary_fracs(sitio)

    assert propuestas, "debe proponer algo"
    for f in propuestas:
        realizadas = fracciones_realizadas(["S1", "S2"], {"S1": f, "S2": 1.0 - f}, n)
        assert realizadas["S1"] == pytest.approx(f, abs=1e-6)
        assert realizadas["S1"] > 0.0 and realizadas["S2"] > 0.0


def test_las_ternarias_de_haluro_reparten_los_24_sitios():
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=3)
    for xI, xBr, xCl in gen._ternary_x_fracs():
        cuentas = cuantizar_ocupacion(["I", "Br", "Cl"], {"I": xI, "Br": xBr, "Cl": xCl}, 24)
        assert sum(n for _, n in cuentas) == 24
        assert all(n > 0 for _, n in cuentas), "las tres especies deben estar presentes"


def test_generar_no_produce_dos_candidatos_con_la_misma_estructura():
    """Era 18 candidatos 'distintos' por cada estructura realmente nueva."""
    from buho.generator.heuristic_generator import HeuristicGenerator

    cands = HeuristicGenerator(str(CONFIG), random_seed=7).generate()
    claves = {clave_estructura(c, [2, 2, 2]) for c in cands}

    assert len(claves) == len(cands), (
        f"{len(cands)} candidatos colapsan en {len(claves)} estructuras"
    )


# ── Deduplicado antes de gastar DFT ──────────────────────────────────────────

def test_no_se_manda_a_dft_una_estructura_ya_calculada(tmp_path, monkeypatch):
    """De 111 filas del CSV real, solo 21 eran estructuras distintas."""
    import pandas as pd

    from buho.discovery.engine import DiscoveryLoop
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    dopado = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.949, "Rb": 0.051})
    puro = _candidato(gen, ["Cs", "Rb"], {"Cs": 1.0, "Rb": 0.0})
    distinto = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.75, "Rb": 0.25})
    candidatos = {c.candidate_id: c for c in (dopado, puro, distinto)}

    loop = DiscoveryLoop(str(CONFIG), models_root=tmp_path)
    monkeypatch.setattr(loop, "_read_ledger", lambda: pd.DataFrame())

    pedidos = [dopado.candidate_id, puro.candidate_id, distinto.candidate_id]
    conservados, descartados = loop._dedup_estructural(pedidos, candidatos)

    assert len(conservados) == 2, "el dopado y el puro son la misma estructura"
    assert distinto.candidate_id in conservados
    assert len(descartados) == 1
    assert descartados[0]["duplica_a"] in conservados


def test_el_descartado_se_registra_con_a_quien_duplica(tmp_path, monkeypatch):
    import pandas as pd

    from buho.discovery.engine import DiscoveryLoop
    from buho.generator.heuristic_generator import HeuristicGenerator

    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    a = _candidato(gen, ["Cs", "Rb"], {"Cs": 0.949, "Rb": 0.051})
    b = _candidato(gen, ["Cs", "Rb"], {"Cs": 1.0, "Rb": 0.0})
    loop = DiscoveryLoop(str(CONFIG), models_root=tmp_path)
    monkeypatch.setattr(loop, "_read_ledger", lambda: pd.DataFrame())

    _, descartados = loop._dedup_estructural([a.candidate_id, b.candidate_id],
                                             {a.candidate_id: a, b.candidate_id: b})
    assert descartados[0]["candidate_id"] == b.candidate_id
    assert descartados[0]["duplica_a"] == a.candidate_id
    assert descartados[0]["formula"] == b.formula


# ── Fuga del target ──────────────────────────────────────────────────────────

def test_el_bandgap_de_dft_no_se_usa_como_feature_para_predecirse(tmp_path):
    """`Eg_target_eV` es `band_gap_gga_eV` + chi_SOC: usarla es copiar el target.

    Daba cv_mae 0.002 eV, y en inferencia la columna no existe —es lo que hay
    que predecir— así que se rellenaba con 0.0 frente a un valor típico de 1.05.
    """
    import pandas as pd

    from buho.discovery.engine import DiscoveryLoop

    filas = []
    for i in range(12):
        eg = 1.0 + 0.01 * i
        filas.append({
            "candidate_id": f"c{i}", "tolerance_t": 0.85 + 0.001 * i,
            "oct_factor": 0.54, "vol_est_A3": 800.0 + i,
            "Eform_eV_atom": -0.3, "band_gap_gga_eV": eg, "Eg_target_eV": eg,
        })
    salida = tmp_path / "data" / "discovery"
    salida.mkdir(parents=True)
    pd.DataFrame(filas).to_csv(salida / "surrogate_training_dft.csv", index=False)

    loop = DiscoveryLoop(str(CONFIG), data_root=tmp_path, models_root=tmp_path)
    loop.training_path = salida / "surrogate_training_dft.csv"
    loop.metrics_path = salida / "model_metrics.jsonl"
    rec = loop._retrain_bandgap(round_id=0)

    assert rec["status"] == "ok"
    modelo = __import__("pickle").load(open(rec["model_path"], "rb"))
    columnas = getattr(modelo, "feature_names", None) or getattr(modelo, "feat_cols", [])
    assert "band_gap_gga_eV" not in list(columnas), (
        f"el target volvio a entrar como feature: {list(columnas)}"
    )


# ── El protocolo autónomo entra por el modo discreto ─────────────────────────

def test_el_modo_discreto_tambien_ajusta_a_la_rejilla():
    """El ajuste solo en el modo continuo dejaba fuera al protocolo autónomo.

    `ChemicalSpaceEnumerator` sobrescribe `generation.fractions` con una rejilla
    de paso 0.0073 y el generador la consume por el camino discreto. Medido: de
    30 000 composiciones salían 1 998 estructuras — 15 candidatos por cada
    estructura nueva, en el camino que gasta el DFT.
    """
    from buho.discovery.space import fraction_grid
    from buho.generator.heuristic_generator import HeuristicGenerator

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["generation"]["fraction_mode"] = "discrete"
    cfg["generation"]["fractions"] = fraction_grid(0.05, 0.95, 0.0073)
    gen = HeuristicGenerator(cfg, random_seed=1)

    for sitio in ("A", "B", "X"):
        n = sitios_por_subred([2, 2, 2])[sitio]
        for f in gen._binary_fracs(sitio):
            realizadas = fracciones_realizadas(
                ["S1", "S2"], {"S1": f, "S2": 1.0 - f}, n)
            assert realizadas["S1"] == pytest.approx(f, abs=1e-6)
            assert realizadas["S1"] > 0.0 and realizadas["S2"] > 0.0


def test_una_rejilla_fina_no_multiplica_los_candidatos():
    """Paso 0.0073 sobre 8 sitios no puede dar más de 7 fracciones distintas."""
    from buho.discovery.space import fraction_grid
    from buho.generator.heuristic_generator import HeuristicGenerator

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["generation"]["fraction_mode"] = "discrete"
    cfg["generation"]["fractions"] = fraction_grid(0.05, 0.95, 0.0073)
    gen = HeuristicGenerator(cfg, random_seed=1)

    propuestas = gen._binary_fracs("A")
    assert len(propuestas) <= 7, f"{len(propuestas)} fracciones para 8 sitios"
    assert propuestas == sorted(set(propuestas)), "sin repetidos"


def test_el_espacio_del_protocolo_no_colapsa(tmp_path):
    """1 candidato = 1 estructura, también por el camino del protocolo."""
    from buho.discovery.space import fraction_grid
    from buho.generator.heuristic_generator import HeuristicGenerator

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["generation"]["fraction_mode"] = "discrete"
    cfg["generation"]["fractions"] = fraction_grid(0.05, 0.95, 0.0073)
    cfg["generation"]["modes"] = {"pure": True, "A_mixed": True, "B_mixed": True,
                                  "X_mixed": False, "multi_mixed": False}
    cands = HeuristicGenerator(cfg, random_seed=3).generate()
    claves = {clave_estructura(c, [2, 2, 2]) for c in cands}

    assert len(claves) == len(cands), (
        f"{len(cands)} candidatos colapsan en {len(claves)} estructuras")


# ── Geometría por pareja B–X ─────────────────────────────────────────────────

#: Parámetros de red experimentales de la fase cúbica, en angstrom.
A_EXPERIMENTAL = {
    ("Pb", "I"): 6.18, ("Pb", "Br"): 5.87, ("Pb", "Cl"): 5.605,
    ("Sn", "I"): 6.22, ("Sn", "Br"): 5.80,
    # Ge: red pseudo-cubica de la fase romboedrica R3m de temperatura ambiente.
    ("Ge", "I"): 5.98, ("Ge", "Br"): 5.63, ("Ge", "Cl"): 5.43,
}


@pytest.mark.parametrize("par", sorted(A_EXPERIMENTAL))
def test_la_celda_reproduce_el_parametro_experimental(cfg, par):
    """La contracción B–X estaba calibrada solo con yoduros.

    Aplicar el factor del yoduro a un bromuro comprimía la celda un 2.1 % y a un
    cloruro un 2.4 %. En estas perovskitas comprimir aumenta el solapamiento
    B–X, sube el máximo de la banda de valencia y cierra el gap: bastaba para
    poner los bromuros por debajo de los yoduros e invertir la tendencia
    Cl > Br > I, que es de las más sólidas de esta familia.
    """
    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    b_site, x_site = par
    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    cand = gen._make_candidate(
        A_sp=["Cs"], B_sp=[b_site], X_sp=[x_site],
        A_f={"Cs": 1.0}, B_f={b_site: 1.0}, X_f={x_site: 1.0}, mode="pure",
    )
    _, meta = ABX3StructureBuilder(cfg).build(cand)

    esperado = A_EXPERIMENTAL[par]
    error = abs(meta["lattice_constant_A"] - esperado) / esperado
    assert error < 0.005, (
        f"Cs{b_site}{x_site}3: a={meta['lattice_constant_A']:.3f} frente a "
        f"{esperado} experimental ({100 * error:+.1f} %)"
    )


def test_una_config_con_el_formato_antiguo_sigue_funcionando(cfg):
    """`{B: factor}` era el formato anterior; una config de usuario escrita
    para esa versión no puede reventar al indexar."""
    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    viejo = dict(cfg)
    viejo["structure"] = dict(cfg["structure"])
    viejo["structure"]["bond_contraction"] = {"Pb": 0.912}
    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    cand = gen._make_candidate(
        A_sp=["Cs"], B_sp=["Pb"], X_sp=["I"],
        A_f={"Cs": 1.0}, B_f={"Pb": 1.0}, X_f={"I": 1.0}, mode="pure",
    )
    _, meta = ABX3StructureBuilder(viejo).build(cand)
    assert meta["lattice_constant_A"] == pytest.approx(6.183, abs=0.01)


def test_el_germanio_se_expande_en_vez_de_contraerse():
    """El radio ionico de Ge2+ supone un cation esferico, pero su par solitario
    4s es estereoquimicamente activo y empuja los haluros mas lejos.

    Sin corregir, CsGeCl3 salia con la celda comprimida un 6.5 % y un gap de
    0.204 eV frente a 3.20 experimental. Con la celda medida sube a 1.277 eV.
    """
    from buho.structure.build_abx3 import BOND_CONTRACTION

    for x_site in ("I", "Br", "Cl"):
        assert BOND_CONTRACTION["Ge"][x_site] > 1.0, (
            f"Ge-{x_site} deberia expandirse, no contraerse")
    # Y el efecto crece al bajar el radio del haluro: cuanto mas pequeno el
    # anion, mas pesa el volumen del par solitario frente al del enlace.
    assert (BOND_CONTRACTION["Ge"]["Cl"]
            > BOND_CONTRACTION["Ge"]["Br"]
            > BOND_CONTRACTION["Ge"]["I"])
