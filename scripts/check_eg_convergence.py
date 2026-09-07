#!/usr/bin/env python3
"""Comprueba si el bandgap del cribado esta convergido en ecut y malla k.

Por que
-------
Con los parametros del cribado (ecut 300 eV, malla 2x2x2) las referencias puras
dan un error medio de -1.09 eV frente al experimento, un 62 % de subestimacion.
PBE subestima, pero tipicamente un 40-50 %, y sobre todo **acierta el orden de
los haluros**: Cl > Br > I. Aqui sale al reves -- CsPbBr3 (0.202 eV con SOC) por
debajo de CsPbI3 (0.400 eV) -- y eso no es un fallo del funcional, es senal de
que el numero no esta convergido.

Este script separa las dos cosas: cuanto se mueve el gap al subir ecut y al
refinar la malla. Si al converger la inversion desaparece, el problema era
numerico y los parametros del cribado hay que subirlos. Si se mantiene, el
problema es el nivel de teoria y hace falta un funcional mejor.

Uso (dentro de WSL, con el python de gpaw246):
    python scripts/check_eg_convergence.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_eg_scale import (  # noqa: E402
    EG_EXPERIMENTAL,
    SMEARING,
    _celda,
    _gap,
    _partir,
)

#: Referencias del barrido: las dos de Pb, que son las que muestran la inversion.
COMPUESTOS = ["CsPbI3", "CsPbBr3"]

#: Barrido: primero ecut a malla fija, luego malla a ecut fijo.
ECUTS = [300, 400, 500, 600]
KMESHES = [2, 4, 6]
ECUT_BASE = 500
KPTS_BASE = 4


def medir(formula: str, ecut: int, k: int) -> dict:
    from gpaw import GPAW, PW, FermiDirac
    from gpaw.spinorbit import soc_eigenstates

    a_site, b_site, x_site = _partir(formula)
    atoms, a = _celda(a_site, b_site, x_site)
    calc = GPAW(
        mode=PW(ecut),
        xc="PBE",
        kpts={"size": [k, k, k], "gamma": True},
        occupations=FermiDirac(SMEARING),
        convergence={"density": 1e-3, "eigenstates": 1e-4, "energy": 1e-4},
        symmetry="off",
        txt=f"conv_{formula}_e{ecut}_k{k}.txt",
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    n_e = int(round(calc.get_number_of_electrons()))
    nk = len(calc.get_ibz_k_points())
    eg_pbe = _gap([calc.get_eigenvalues(kpt=i) for i in range(nk)], n_e // 2)
    eg_soc = _gap(soc_eigenstates(calc).eigenvalues(), n_e)
    return {"formula": formula, "ecut": ecut, "kpts": k,
            "Eg_pbe_eV": round(eg_pbe, 4), "Eg_pbe_soc_eV": round(eg_soc, 4),
            "Eg_exp_eV": EG_EXPERIMENTAL[formula]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/eg_convergence.json"))
    args = ap.parse_args()

    filas = []
    print("=== ecut, con malla fija", KPTS_BASE, "===", flush=True)
    for formula in COMPUESTOS:
        for ecut in ECUTS:
            r = medir(formula, ecut, KPTS_BASE)
            filas.append(r)
            print(f"  {formula:8s} ecut={ecut:4d} k={KPTS_BASE}  "
                  f"PBE={r['Eg_pbe_eV']:6.3f}  +SOC={r['Eg_pbe_soc_eV']:6.3f}", flush=True)

    print("=== malla k, con ecut fijo", ECUT_BASE, "===", flush=True)
    for formula in COMPUESTOS:
        for k in KMESHES:
            if k == KPTS_BASE:
                continue
            r = medir(formula, ECUT_BASE, k)
            filas.append(r)
            print(f"  {formula:8s} ecut={ECUT_BASE} k={k}  "
                  f"PBE={r['Eg_pbe_eV']:6.3f}  +SOC={r['Eg_pbe_soc_eV']:6.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(filas, indent=2), encoding="utf-8")
    print(f"\nEscrito {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
