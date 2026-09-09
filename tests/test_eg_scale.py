"""Escala del bandgap: del valor calculado al medido.

El cribado predice en la escala del cálculo (PBE + SOC perturbativo) y compara
contra la ventana fotovoltaica, que está definida sobre el gap medido. Medido
sobre el protocolo autónomo, la diferencia no era gradual sino total: de 3793
candidatos, los 3793 caían en Tier 1 porque el surrogate predecía ~0.96 eV y la
ventana empieza en 1.1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buho import eg_scale  # noqa: E402

AJUSTE = {
    "columnas": ["c", "B:Pb", "B:Sn", "X:Br", "X:Cl"],
    "coeficientes": [1.0, 0.3, -0.2, 0.5, 0.9],
}


@pytest.fixture(autouse=True)
def _sin_cache():
    eg_scale._cache.clear()
    yield
    eg_scale._cache.clear()


# ── El desplazamiento ────────────────────────────────────────────────────────

def test_el_desplazamiento_depende_del_haluro_no_solo_del_sitio_B():
    """Suponerlo constante por B sería falso: medido, CsPbI3 y CsPbBr3 difieren
    en 0.83 eV con el mismo elemento B."""
    yoduro = eg_scale.delta({"Pb": 1.0}, {"I": 1.0}, AJUSTE)
    bromuro = eg_scale.delta({"Pb": 1.0}, {"Br": 1.0}, AJUSTE)
    assert yoduro != bromuro
    assert bromuro - yoduro == pytest.approx(0.5)


def test_una_composicion_mixta_interpola():
    mezcla = eg_scale.delta({"Pb": 1.0}, {"I": 0.5, "Br": 0.5}, AJUSTE)
    puros = [eg_scale.delta({"Pb": 1.0}, {x: 1.0}, AJUSTE) for x in ("I", "Br")]
    assert mezcla == pytest.approx(sum(puros) / 2)


def test_un_elemento_sin_calibrar_aporta_cero():
    """Corregir de menos es preferible a inventar el valor."""
    con_ge = eg_scale.delta({"Ge": 1.0}, {"I": 1.0}, AJUSTE)
    assert con_ge == pytest.approx(1.0), "solo queda el término constante"


def test_sin_tabla_no_se_toca_el_valor():
    """Un pipeline sin calibrar debe comportarse como antes, no romperse."""
    assert eg_scale.a_experimental(1.23, {"Pb": 1.0}, {"I": 1.0}, {}) == 1.23
    assert eg_scale.delta({"Pb": 1.0}, {"I": 1.0}, {}) == 0.0


def test_none_entra_y_sale_none():
    """Que el surrogate no prediga nada no es lo mismo que predecir cero."""
    assert eg_scale.a_experimental(None, {"Pb": 1.0}, {"I": 1.0}, AJUSTE) is None


def test_la_correccion_sube_el_valor(tmp_path):
    """PBE subestima y el SOC hunde más: el desplazamiento es positivo."""
    tabla = tmp_path / "config" / "eg_scale.json"
    tabla.parent.mkdir(parents=True)
    tabla.write_text(json.dumps({"ajuste": AJUSTE}), encoding="utf-8")
    cargado = eg_scale.cargar_tabla(tabla)

    corregido = eg_scale.a_experimental(0.4, {"Pb": 1.0}, {"I": 1.0}, cargado)
    assert corregido > 0.4


# ── Carga de la tabla ────────────────────────────────────────────────────────

def test_una_tabla_ausente_avisa_en_vez_de_callarse(tmp_path, caplog):
    """Sin corrección, la ventana PV se aplica sobre una magnitud que no es la
    suya y descarta candidatos válidos. Es el fallo que dejó el scissor de SOC
    sin aplicarse en todos los binarios publicados."""
    with caplog.at_level("WARNING"):
        tabla = eg_scale.cargar_tabla(tmp_path / "no-existe.json")

    assert tabla == {}
    assert "escala" in caplog.text.lower()


def test_una_tabla_ilegible_no_tumba_el_cribado(tmp_path, caplog):
    roto = tmp_path / "eg_scale.json"
    roto.write_text("{no es json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert eg_scale.cargar_tabla(roto) == {}
    assert "ilegible" in caplog.text


def test_el_binario_congelado_busca_primero_en_el_bundle(monkeypatch):
    """`__file__` cuelga del directorio de extracción, así que `parents[2]`
    apunta por encima del bundle. Es exactamente el fallo que dejó la tabla de
    SOC sin cargar en los binarios publicados."""
    monkeypatch.setattr(eg_scale.sys, "frozen", True, raising=False)
    monkeypatch.setattr(eg_scale.sys, "_MEIPASS", "/bundle", raising=False)

    raices = eg_scale._raices()

    assert raices[0] == Path("/bundle")


def test_describir_dice_si_hay_calibracion():
    assert eg_scale.describir({})["disponible"] is False
    assert eg_scale.describir(AJUSTE)["disponible"] is True


# ── El sello de escala ───────────────────────────────────────────────────────

class _ModeloFalso:
    pass


def test_un_modelo_sin_sello_no_se_corrige():
    """Corregir un modelo entrenado en otra escala no es corregir a medias.

    Pasó de verdad: el modelo publicado predecía ~0.96 eV —etiquetas de PBE
    sobre la geometría vieja, sin SOC— y sumarle el desplazamiento lo llevaba a
    3.04 eV. Antes caía fuera de la ventana por abajo; después seguía fuera, por
    arriba, y con aspecto de estar corregido.
    """
    assert eg_scale.aplicable_a(_ModeloFalso()) is False
    assert eg_scale.escala_del_modelo(_ModeloFalso()) is None


def test_un_modelo_recien_entrenado_lleva_su_escala():
    modelo = eg_scale.sellar_modelo(_ModeloFalso())

    assert eg_scale.escala_del_modelo(modelo) == eg_scale.ESCALA
    assert eg_scale.aplicable_a(modelo) is True


def test_una_escala_distinta_no_se_da_por_buena():
    modelo = _ModeloFalso()
    setattr(modelo, eg_scale.ATRIBUTO_MODELO, "otra_escala_vieja")

    assert eg_scale.aplicable_a(modelo) is False


# ── El error de la propia calibracion ────────────────────────────────────────
#
# La tabla mide su error (6.0 % sobre los nueve haluros de referencia) y hasta
# ahora lo tiraba al parsear. La ventana del Tier 1 contaba la sigma del
# surrogate y daba el desplazamiento por exacto, asi que descartaba candidatos
# validos por menos de lo que la propia tabla declaraba equivocarse.

def test_el_error_medido_viaja_con_el_ajuste(tmp_path):
    tabla = tmp_path / "eg_scale.json"
    tabla.write_text(json.dumps({
        "ajuste": AJUSTE,
        "errores_pct": {"error_relativo_medio_sin_corregir": 84.5,
                        "error_relativo_medio_corregido": 6.0},
    }), encoding="utf-8")

    cargado = eg_scale.cargar_tabla(tabla)

    assert cargado["error_relativo_pct"] == 6.0
    assert eg_scale.describir(cargado)["error_relativo_pct"] == 6.0


def test_el_margen_es_relativo_al_gap():
    """El error de la tabla es relativo, asi que en eV crece con el gap. Un
    margen fijo seria demasiado en el borde inferior y demasiado poco en el
    superior, que es donde se perdian los candidatos."""
    ajuste = {**AJUSTE, "error_relativo_pct": 6.0}

    assert eg_scale.margen_calibracion(1.80, ajuste) == pytest.approx(0.108)
    assert eg_scale.margen_calibracion(1.10, ajuste) == pytest.approx(0.066)


def test_sin_tabla_no_hay_margen_que_contar():
    """Sin correccion de escala tampoco hay error de correccion, y el numero
    cribado ni siquiera esta en esa escala."""
    assert eg_scale.margen_calibracion(1.8, {}) == 0.0
    assert eg_scale.margen_calibracion(1.8, AJUSTE) == 0.0
    assert eg_scale.margen_calibracion(None, {**AJUSTE, "error_relativo_pct": 6.0}) == 0.0


def test_un_error_no_numerico_avisa_y_no_tumba_la_carga(tmp_path, caplog):
    tabla = tmp_path / "eg_scale.json"
    tabla.write_text(json.dumps({
        "ajuste": AJUSTE,
        "errores_pct": {"error_relativo_medio_corregido": "seis por ciento"},
    }), encoding="utf-8")

    with caplog.at_level("WARNING"):
        cargado = eg_scale.cargar_tabla(tabla)

    assert cargado["coeficientes"]          # el ajuste sigue sirviendo
    assert "error_relativo_medio_corregido" in caplog.text
    assert eg_scale.margen_calibracion(1.8, cargado) == 0.0


@pytest.mark.parametrize("eg, exp, nombre", [
    (1.839, 1.73, "CsPbI3"),
    (1.844, 1.75, "CsSnBr3"),
])
def test_los_dos_falsos_negativos_de_borde_vuelven_a_la_ventana(eg, exp, nombre):
    """Los dos casos que la auditoria midio: predichos justo por encima de 1.8,
    con el valor experimental dentro. Con el margen de la calibracion dejan de
    caer fuera; sin el, se perdian por centesimas."""
    ajuste = {**AJUSTE, "error_relativo_pct": 6.0}
    margen = eg_scale.margen_calibracion(eg, ajuste)

    assert eg > 1.8, f"{nombre}: el caso solo existe si el predicho se sale"
    assert exp < 1.8
    assert eg - margen < 1.8, f"{nombre} seguiria descartandose"
