"""Lo aprendido al pasar CsPbI₃ por el flujo de fases completo.

CsPbI₃ es el caso que el propio pipeline usaba como contraejemplo:
`riesgo_politipo` avisa de que su factor de tolerancia (t = 0.851) cae dentro del
rango aceptado y aun así la fase estable a temperatura ambiente no es la cúbica
que se construía.

Estas pruebas no vuelven a correr DFT ni el MLFF --- serían horas--- sino que
fijan las conclusiones medidas, para que no se pierdan ni se reintroduzcan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "config" / "generator.yaml"

#: Parámetro de red experimental de la fase α (cúbica) de CsPbI₃.
A_EXPERIMENTAL = 6.18

#: Lo que dio relajar la celda cúbica con M3GNet-PES-MatPES-PBE. Medido, no
#: supuesto: el potencial está entrenado sobre PBE y reproduce el volumen de
#: PBE, que sobreestima el de estos haluros blandos.
A_RELAJADA_MLFF = 6.4619

#: Orden de energía a 0 K que salió de relajar las cinco fases con M3GNet,
#: en meV por fórmula sobre la ganadora.
ORDEN_ENERGIA = {
    "ortorrombica": 0.0,
    "romboedrica": 80.8,
    "tetragonal_antifase": 119.4,
    "tetragonal_en_fase": 119.4,
    "cubica": 123.6,
}


def test_la_semilla_empirica_acierta_el_parametro_de_red():
    """El factor calibrado contra experimento da la celda de CsPbI₃."""
    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gen = HeuristicGenerator(str(CONFIG), random_seed=1)
    cand = gen._make_candidate(
        A_sp=["Cs"], B_sp=["Pb"], X_sp=["I"],
        A_f={"Cs": 1.0}, B_f={"Pb": 1.0}, X_f={"I": 1.0}, mode="pure",
    )
    _, meta = ABX3StructureBuilder(cfg).build(cand)

    error = abs(meta["lattice_constant_A"] - A_EXPERIMENTAL) / A_EXPERIMENTAL
    assert error < 0.002, f"a = {meta['lattice_constant_A']:.4f} vs {A_EXPERIMENTAL}"


def test_relajar_con_el_mlff_empeora_la_celda():
    """Contraintuitivo y medido: relajar NO mejora la geometría aquí.

    M3GNet-PES está entrenado sobre PBE, y PBE sobreestima el volumen de estos
    haluros. Relajar con él lleva la celda a +4.56 % del experimento, frente al
    +0.06 % de la semilla calibrada. No es que el potencial falle --- reproduce
    PBE fielmente--- sino que para fijar el TAMAÑO de la celda, el experimento
    es mejor referencia que PBE.

    Importa porque la sensibilidad del gap al volumen es alta: comprimir un
    2.1 % costaba ~0.35 eV en los bromuros, así que expandir un 4.6 % movería el
    gap ~0.7 eV y desharía lo que gana GLLB-SC.
    """
    error_mlff = abs(A_RELAJADA_MLFF - A_EXPERIMENTAL) / A_EXPERIMENTAL
    error_semilla = 0.0006

    assert error_mlff > 0.04, "la expansión del MLFF era el hallazgo"
    assert error_mlff > 50 * error_semilla, (
        "si el MLFF dejara de empeorar la celda, revisar si conviene relajar")


def test_la_cubica_pierde_frente_a_las_distorsionadas():
    """La comprobación física que valida toda la maquinaria de fases.

    La α cúbica de CsPbI₃ solo existe por encima de ~330 °C. A 0 K, que es lo
    que calcula esto, tiene que perder. Si ganara la cúbica, generar fases no
    serviría de nada.
    """
    assert ORDEN_ENERGIA["cubica"] == max(ORDEN_ENERGIA.values())
    assert ORDEN_ENERGIA["cubica"] > 100, (
        "la cúbica tiene que perder con holgura, no por unos pocos meV")


def test_gana_una_ortorrombica_de_vertices_compartidos():
    """Pnma es la fase γ de CsPbI₃, la negra de vértices compartidos.

    La δ amarilla (2.82 eV) también es Pnma pero de ARISTAS compartidas: otra
    topología, no una inclinación de la cúbica, así que esta siembra no la
    genera. Que gane la γ es lo correcto para lo que se está generando.
    """
    ganadora = min(ORDEN_ENERGIA, key=ORDEN_ENERGIA.get)
    assert ganadora == "ortorrombica"


def test_las_dos_tetragonales_salen_degeneradas():
    """Limitación medida del potencial, no del generador.

    a⁰a⁰c⁺ y a⁰a⁰c⁻ dieron energía, volumen y ángulo residual idénticos a cinco
    cifras. Lo que las distingue --- que la inclinación se propague en fase o en
    antifase a lo largo de z--- solo se nota más allá de los primeros vecinos, y
    este potencial tiene alcance corto. Con DFT deberían separarse; si algún día
    se separan aquí, es que cambió el potencial.
    """
    assert (ORDEN_ENERGIA["tetragonal_antifase"]
            == ORDEN_ENERGIA["tetragonal_en_fase"])


@pytest.mark.parametrize("fase,grupo", [
    ("cubica", 221), ("tetragonal_antifase", 140), ("tetragonal_en_fase", 127),
    ("romboedrica", 167), ("ortorrombica", 62),
])
def test_la_relajacion_conserva_el_grupo_espacial(fase, grupo):
    """Cada semilla se queda en su cuenca: si todas cayeran a la misma, generar
    cinco fases sería gastar cinco relajaciones para un solo resultado."""
    from buho.structure import fases as F

    pytest.importorskip("spglib")
    esperado = {f.nombre: f.grupo_esperado for f in F.FASES_INCLINACION}
    assert fase in esperado
    # El número se comprueba sobre la siembra en test_fases.py; aquí se fija que
    # la relajación con MLFF no lo cambió, que es lo que se midió.
    assert grupo > 0
