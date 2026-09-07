#!/usr/bin/env python3
"""Calibra el desplazamiento de escala entre el bandgap calculado y el medido.

Por que hace falta
------------------
El cribado predice un bandgap en la escala del calculo (PBE con SOC
perturbativo) y lo compara contra la ventana fotovoltaica [1.1, 1.8] eV, que
sale del limite de Shockley-Queisser y esta en escala **experimental**. Son
magnitudes distintas, y la diferencia no es pequena.

Medido sobre el protocolo autonomo: de 3793 candidatos, los 3793 caian en el
Tier 1 porque el surrogate predecia entre 0.87 y 1.04 eV y la ventana empieza en
1.1. El bucle cribaba, no encontraba nada elegible y se declaraba terminado sin
lanzar un solo DFT. En el propio conjunto de entrenamiento, solo 2 de 111 filas
caian dentro de la ventana contra la que se las juzgaba.

El problema tiene dos mitades que empujan en sentidos opuestos:

  * PBE subestima el gap por el error de autointeraccion y la discontinuidad de
    derivada que le falta al funcional. Para estos haluros, del orden de +0.5 a
    +1 eV por debajo del valor real.
  * El SOC, que si se aplica, lo hunde todavia mas: para Pb vale casi -0.7 eV.

Sumadas, el calculo queda muy por debajo del experimento. Este script mide esa
diferencia contra compuestos con valor experimental establecido, en vez de
suponer un numero.

Metodo
------
Para cada referencia construye la celda cubica con la misma geometria que usa el
pipeline, corre el mismo SCF que el cribado, extrae el gap con SOC y lo compara
con el valor experimental publicado. El desplazamiento se agrega por elemento
del sitio B, porque las dos mitades del error dependen de B: la subestimacion de
PBE va con el caracter del CBM y el SOC crece con el numero atomico.

Uso (dentro de WSL, con el python de gpaw246):
    python scripts/calibrate_eg_scale.py --out config/eg_scale.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

#: Bandgaps experimentales de referencia, en eV, de la fase que el pipeline
#: modela (perovskita cubica de vertices compartidos).
#:
#: Son ENTRADAS de la calibracion, no resultados: se listan aqui para que se
#: puedan auditar y cambiar sin tocar el codigo. Cada uno corresponde a la fase
#: indicada; usar el valor de otra fase desplazaria la calibracion entera.
#:
#:   CsPbI3   1.73  fase alfa (cubica negra), la que modela el pipeline.
#:                  La fase delta amarilla a temperatura ambiente tiene ~2.8 eV
#:                  y NO es la que se calcula.
#:   CsPbBr3  2.36  fase cubica de alta temperatura.
#:   CsSnI3   1.31  fase B-gamma (perovskita negra).
#:   CsGeI3   1.63  romboedrica a temperatura ambiente; se modela como cubica,
#:                  asi que este es el punto menos firme de la tabla.
#: La rejilla 3x3 de haluros de Cs. Cuatro puntos no bastan para separar el
#: efecto del sitio B del efecto del haluro, y medido resulta que el
#: desplazamiento depende de los dos: entre CsPbI3 y CsPbBr3 hay 0.83 eV de
#: diferencia en el mismo elemento B.
#:
#: `firme` marca si el compuesto cristaliza de verdad en la perovskita cubica
#: que el pipeline modela. Los que no, entran en el ajuste con menos peso: su
#: valor experimental corresponde a otra estructura y arrastraria la
#: calibracion.
EG_EXPERIMENTAL = {
    # B = Pb: los mejor establecidos de la familia.
    "CsPbI3":  {"eg": 1.73, "firme": True,
                "fase": "alfa cubica (negra); la delta amarilla de RT tiene ~2.8 eV"},
    "CsPbBr3": {"eg": 2.36, "firme": True, "fase": "cubica de alta temperatura"},
    "CsPbCl3": {"eg": 3.03, "firme": True, "fase": "cubica de alta temperatura"},
    # B = Sn.
    "CsSnI3":  {"eg": 1.31, "firme": True, "fase": "B-gamma (perovskita negra)"},
    "CsSnBr3": {"eg": 1.75, "firme": True, "fase": "cubica"},
    "CsSnCl3": {"eg": 2.80, "firme": False,
                "fase": "monoclinica a RT; el valor no es de la fase cubica"},
    # B = Ge: los octaedros salen muy distorsionados por el par solitario 4s,
    # asi que la cubica ideal es peor aproximacion que en Pb/Sn.
    "CsGeI3":  {"eg": 1.63, "firme": False, "fase": "romboedrica a RT"},
    "CsGeBr3": {"eg": 2.32, "firme": False, "fase": "romboedrica a RT"},
    "CsGeCl3": {"eg": 3.20, "firme": False,
                "fase": "romboedrica a RT; valor con dispersion en la literatura"},
}

#: Radios ionicos (A). Coordinacion 12 para A, 6 para B y X. Identicos a los
#: del pipeline: calibrar con otra geometria daria un desplazamiento que no
#: corresponde a las celdas sobre las que se va a aplicar.
RADII = {"Cs": 1.88, "Pb": 1.19, "Sn": 1.18, "Ge": 0.73, "I": 2.20,
         "Br": 1.96, "Cl": 1.81}

#: La misma contraccion que usa el constructor de estructuras
#: (buho.structure.build_abx3.BOND_CONTRACTION), por pareja B-X. Calibrar con
#: otra geometria daria un desplazamiento que no corresponde a las celdas sobre
#: las que se va a aplicar.
BOND_CONTRACTION = {
    "Pb": {"I": 0.912, "Br": 0.932, "Cl": 0.934},
    "Sn": {"I": 0.920, "Br": 0.924, "Cl": 0.931},
}

#: Parametros del calculo. Por defecto los del cribado; se pueden subir desde
#: la linea de ordenes para calibrar sobre numeros convergidos, que es lo unico
#: que tiene sentido: un desplazamiento ajustado sobre gaps sin converger
#: absorbe el error numerico y deja de valer al cambiar los parametros.
ECUT = 300
KPTS = [2, 2, 2]
SMEARING = 0.01

#: Peso en el ajuste de los compuestos cuya fase real no es la cubica que se
#: modela. No se descartan —informan— pero no deben arrastrar la calibracion.
PESO_NO_FIRME = 0.3


def _partir(formula: str) -> tuple[str, str, str]:
    """`CsPbI3` -> (Cs, Pb, I). Solo compuestos puros ABX3."""
    for a in ("Cs", "Rb", "K"):
        if formula.startswith(a):
            resto = formula[len(a):]
            break
    else:
        raise ValueError(f"Sitio A no reconocido en {formula}")
    for b in ("Pb", "Sn", "Ge"):
        if resto.startswith(b):
            x = resto[len(b):].rstrip("3")
            return a, b, x
    raise ValueError(f"Sitio B no reconocido en {formula}")


def _celda(a_site: str, b_site: str, x_site: str):
    from ase import Atoms

    a = 2.0 * (RADII[b_site] + RADII[x_site]) * \
        BOND_CONTRACTION.get(b_site, {}).get(x_site, 1.0)
    return Atoms(
        symbols=[a_site, b_site, x_site, x_site, x_site],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    ), a


def _gap(eigenvalues, n_ocupados: int) -> float:
    vbm = max(e[n_ocupados - 1] for e in eigenvalues)
    cbm = min(e[n_ocupados] for e in eigenvalues)
    return float(cbm - vbm)


def medir(formula: str, *, ecut: int = ECUT, kpts: list[int] | None = None,
          verbose: bool = True) -> dict:
    """Gap calculado (PBE y PBE+SOC) de una referencia pura."""
    from gpaw import GPAW, PW, FermiDirac
    from gpaw.spinorbit import soc_eigenstates

    kpts = kpts or KPTS
    a_site, b_site, x_site = _partir(formula)
    atoms, a = _celda(a_site, b_site, x_site)
    if verbose:
        print(f"  {formula}  a = {a:.4f} A  ecut={ecut} k={kpts[0]}", flush=True)

    calc = GPAW(
        mode=PW(ecut),
        xc="PBE",
        kpts={"size": kpts, "gamma": True},
        occupations=FermiDirac(SMEARING),
        convergence={"density": 1e-3, "eigenstates": 1e-4, "energy": 1e-4},
        symmetry="off",
        txt=f"eg_scale_{formula}.txt",
    )
    atoms.calc = calc
    atoms.get_potential_energy()

    n_e = int(round(calc.get_number_of_electrons()))
    nk = len(calc.get_ibz_k_points())
    eg_pbe = _gap([calc.get_eigenvalues(kpt=k) for k in range(nk)], n_e // 2)
    soc = soc_eigenstates(calc)
    eg_soc = _gap(soc.eigenvalues(), n_e)

    ref = EG_EXPERIMENTAL[formula]
    return {
        "formula": formula,
        "B": b_site,
        "X": x_site,
        "a_lat_A": round(a, 4),
        "ecut": ecut,
        "kpts": list(kpts),
        "Eg_pbe_eV": round(eg_pbe, 4),
        "Eg_pbe_soc_eV": round(eg_soc, 4),
        "Eg_exp_eV": ref["eg"],
        "fase_firme": ref["firme"],
        "fase": ref["fase"],
        # Lo que habria que SUMAR al gap con SOC para llegar al experimental.
        "residuo_eV": round(ref["eg"] - eg_soc, 4),
    }


def ajustar(detalle: list[dict]) -> dict:
    """Ajusta el residuo como una suma de contribuciones de B y de X.

    Eg_exp - Eg_calc  ~=  c + delta_B + delta_X

    Se elige esta forma y no un desplazamiento por elemento B porque medido no
    lo es: entre CsPbI3 y CsPbBr3, con el mismo B, hay 0.83 eV de diferencia. Y
    no se ajusta algo mas flexible porque con nueve puntos cualquier modelo con
    mas parametros memoriza en vez de generalizar.

    Calibracion (gauge): delta_X del yoduro se fija a 0, porque de otro modo c y
    los delta no son separables. Solo importan las diferencias.
    """
    import numpy as np

    bs = sorted({d["B"] for d in detalle})
    xs = sorted({d["X"] for d in detalle})
    ancla = "I" if "I" in xs else xs[0]
    libres_x = [x for x in xs if x != ancla]
    cols = ["c"] + [f"B:{b}" for b in bs[1:]] + [f"X:{x}" for x in libres_x]

    A, y, w = [], [], []
    for d in detalle:
        fila = [1.0]
        fila += [1.0 if d["B"] == b else 0.0 for b in bs[1:]]
        fila += [1.0 if d["X"] == x else 0.0 for x in libres_x]
        A.append(fila)
        y.append(d["residuo_eV"])
        w.append(1.0 if d["fase_firme"] else PESO_NO_FIRME)

    A = np.asarray(A, float)
    y = np.asarray(y, float)
    raiz_w = np.sqrt(np.asarray(w, float))
    coef, *_ = np.linalg.lstsq(A * raiz_w[:, None], y * raiz_w, rcond=None)
    return {"columnas": cols, "coeficientes": [round(float(v), 4) for v in coef],
            "ancla_X": ancla, "B_base": bs[0]}


def aplicar(ajuste: dict, b_site: str, x_site: str) -> float:
    """El desplazamiento que corresponde a una pareja (B, X)."""
    cols, coef = ajuste["columnas"], ajuste["coeficientes"]
    total = 0.0
    for nombre, v in zip(cols, coef):
        if nombre == "c":
            total += v
        elif nombre == f"B:{b_site}" or nombre == f"X:{x_site}":
            total += v
    return total


def _errores(detalle: list[dict], ajuste: dict | None) -> dict:
    """Error medio relativo antes y despues de corregir."""
    crudo, corregido = [], []
    for d in detalle:
        exp = d["Eg_exp_eV"]
        crudo.append(abs(d["Eg_pbe_soc_eV"] - exp) / exp)
        if ajuste is not None:
            pred = d["Eg_pbe_soc_eV"] + aplicar(ajuste, d["B"], d["X"])
            corregido.append(abs(pred - exp) / exp)
    n = len(detalle)
    out = {"error_relativo_medio_sin_corregir": round(100 * sum(crudo) / n, 1)}
    if corregido:
        out["error_relativo_medio_corregido"] = round(100 * sum(corregido) / n, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("config/eg_scale.json"))
    ap.add_argument("--solo", nargs="*", help="Limitar a estas referencias.")
    ap.add_argument("--ecut", type=int, default=ECUT)
    ap.add_argument("--kpts", type=int, default=KPTS[0],
                    help="Malla k n x n x n, gamma-centrada.")
    args = ap.parse_args()

    referencias = args.solo or list(EG_EXPERIMENTAL)
    detalle = []
    for formula in referencias:
        if formula not in EG_EXPERIMENTAL:
            print(f"Sin valor experimental para {formula}", file=sys.stderr)
            return 2
        detalle.append(medir(formula, ecut=args.ecut,
                            kpts=[args.kpts] * 3))

    ajuste = ajustar(detalle) if len(detalle) >= 4 else None
    errores = _errores(detalle, ajuste)

    salida = {
        "generado": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metodo": (
            f"SCF PBE (ecut={args.ecut} eV, kpts={args.kpts}^3 gamma-centrada) "
            "sobre la perovskita cubica de referencia, SOC perturbativo "
            "post-SCF, comparado con el bandgap experimental de esa fase."
        ),
        "modelo": (
            "Eg_exp ~= Eg_pbe_soc + c + delta_B + delta_X. Se ajusta por minimos "
            "cuadrados con los compuestos de fase no cubica pesados a "
            f"{PESO_NO_FIRME}. El desplazamiento NO es una constante por elemento "
            "B: medido, CsPbI3 y CsPbBr3 difieren en 0.83 eV con el mismo B."
        ),
        "aviso": (
            "Calibrado sobre haluros puros de Cs. Aplicarlo a composiciones "
            "mixtas supone que el desplazamiento interpola linealmente con las "
            "fracciones de cada sitio, que es una aproximacion sin comprobar."
        ),
        "ajuste": ajuste,
        "errores_pct": errores,
        "detalle": detalle,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(salida, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    print(f"\nEscrito {args.out}")
    print(f"  error relativo medio sin corregir: "
          f"{errores['error_relativo_medio_sin_corregir']} %")
    if "error_relativo_medio_corregido" in errores:
        print(f"  error relativo medio corregido   : "
              f"{errores['error_relativo_medio_corregido']} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
