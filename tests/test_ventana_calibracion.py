"""La ventana del Tier 1 cuenta el error de la calibracion, no solo el del modelo.

Hasta ahora la malla contaba la sigma del surrogate y daba por exacto el
desplazamiento que lleva el gap a escala experimental. Pero ese desplazamiento
tiene su propio error --- 6.0 % medido sobre los nueve haluros de referencia,
~0.11 eV en el borde superior de la ventana--- y es el que manda: la auditoria
midio dos candidatos validos descartados por centesimas, CsPbI3 (1.839 predicho
frente a 1.73 medido) y CsSnBr3 (1.844 / 1.75).

Estas pruebas ejercitan `screen()` de verdad, no el predicado reimplementado:
que el margen exista en `eg_scale` no sirve de nada si el gate no lo suma.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buho import eg_scale  # noqa: E402
from buho.generator.heuristic_generator import HeuristicGenerator  # noqa: E402
from buho.screening import cascade as mod_cascade  # noqa: E402
from buho.screening.cascade import ScreeningCascade  # noqa: E402

CONFIG = ROOT / "config" / "generator.yaml"

#: Ajuste neutro: desplazamiento 0 para que el Eg que se criba sea exactamente
#: el que predice el doble, y 6.0 % de error para medir solo el margen.
AJUSTE_NEUTRO = {"columnas": ["c"], "coeficientes": [0.0], "error_relativo_pct": 6.0}


class _SurrogateFalso:
    """Predice lo que se le diga y declara la escala que la calibracion asume.

    Sin el sello, `cascade` no aplica la correccion --- y entonces tampoco debe
    aplicar el margen, porque el numero no esta en esa escala.
    """

    def __init__(self, eg: float, sigma: float, *, sellado: bool = True):
        self.feature_cols = ["tolerance_t"]
        self._eg = eg
        self._sigma = sigma
        if sellado:
            setattr(self, eg_scale.ATRIBUTO_MODELO, eg_scale.ESCALA)

    def predict_batch(self, X):
        n = len(X)
        return np.full(n, self._eg), np.full(n, self._sigma)


@pytest.fixture(autouse=True)
def _sin_cache():
    eg_scale._cache.clear()
    yield
    eg_scale._cache.clear()


@pytest.fixture
def candidatos():
    if not CONFIG.is_file():
        pytest.skip("config/generator.yaml no disponible")
    g = HeuristicGenerator(str(CONFIG))
    batch = [c for c in g.generate() if c.generation_mode == "pure"][:6]
    assert batch, "hacen falta candidatos puros que pasen el Tier 0"
    return batch


def _cribar(candidatos, monkeypatch, *, eg, sigma, ajuste, sellado=True):
    monkeypatch.setattr(eg_scale, "cargar_tabla", lambda *a, **k: ajuste)
    monkeypatch.setattr(mod_cascade, "build_X",
                        lambda df, cols, **k: np.zeros((len(df), 1)))

    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.setdefault("screening", {}).update({"sigma_k": 1.0, "tier2_mlff": False})
    casc = ScreeningCascade(cfg, project_root=ROOT)
    casc._surrogate = _SurrogateFalso(eg, sigma, sellado=sellado)

    return casc.screen(candidatos, run_mlff=False)


@pytest.mark.parametrize("eg, medido, nombre", [
    (1.839, 1.73, "CsPbI3"),
    (1.844, 1.75, "CsSnBr3"),
])
def test_el_borde_superior_sobrevive_con_el_error_de_la_tabla(
        candidatos, monkeypatch, eg, medido, nombre):
    """Los dos casos que la auditoria midio. Sigma del modelo a cero para que
    lo unico que pueda salvarlos sea el margen de la calibracion."""
    assert medido < 1.8 < eg, f"{nombre}: el caso solo existe si el predicho se sale"

    con = _cribar(candidatos, monkeypatch, eg=eg, sigma=0.0, ajuste=AJUSTE_NEUTRO)
    sin = _cribar(candidatos, monkeypatch, eg=eg, sigma=0.0,
                  ajuste={k: v for k, v in AJUSTE_NEUTRO.items()
                          if k != "error_relativo_pct"})

    assert (con["dropped_at_tier"] != 1).all(), f"{nombre} deberia sobrevivir"
    assert (sin["dropped_at_tier"] == 1).all(), (
        f"{nombre} sin margen de calibracion se pierde por centesimas")


def test_lo_que_esta_lejos_se_sigue_descartando(candidatos, monkeypatch):
    """El margen ensancha la duda, no la ventana: 3.0 eV con 6 % son 0.18, y
    sigue estando a mas de un eV de 1.8."""
    df = _cribar(candidatos, monkeypatch, eg=3.0, sigma=0.0, ajuste=AJUSTE_NEUTRO)

    assert (df["dropped_at_tier"] == 1).all()


def test_el_motivo_dice_cuanto_puso_cada_error(candidatos, monkeypatch):
    """La costumbre del proyecto es que el mensaje diga que decidio: con dos
    margenes sumados, uno solo no permite reconstruir el criterio."""
    df = _cribar(candidatos, monkeypatch, eg=3.0, sigma=0.2, ajuste=AJUSTE_NEUTRO)

    motivo = df["drop_reason"].iloc[0]
    assert "0.20" in motivo, f"falta el margen del modelo en: {motivo}"
    assert "0.18" in motivo, f"falta el margen de la calibracion en: {motivo}"
    assert "calibracion" in motivo


def test_sin_sello_no_se_suma_un_margen_que_no_significa_nada(candidatos, monkeypatch):
    """Si el modelo no declara la escala, `eg` no esta en escala experimental:
    su error de calibracion no aplica y el gate vuelve a ser solo sigma."""
    df = _cribar(candidatos, monkeypatch, eg=1.839, sigma=0.0,
                 ajuste=AJUSTE_NEUTRO, sellado=False)

    assert (df["dropped_at_tier"] == 1).all()
