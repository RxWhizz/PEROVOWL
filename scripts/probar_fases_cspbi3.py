#!/usr/bin/env python3
"""Flujo completo de fases sobre CsPbI3, contra lo medido y la literatura.

Por que CsPbI3
--------------
Es el caso que el propio pipeline usaba como contraejemplo: `riesgo_politipo`
advierte que su factor de tolerancia (t = 0.851) cae dentro del rango aceptado y
sin embargo la fase estable a temperatura ambiente NO es la cubica que el
pipeline construia. Ademas ya tenemos numeros propios contra los que contrastar.

Lo que comprueba
----------------
1. Que las cinco fases se siembran con el grupo espacial correcto.
2. Que al relajar con M3GNet la CUBICA NO gana. Es la prediccion fisica que
   importa: la alfa cubica de CsPbI3 solo existe por encima de ~330 C, y a 0 K
   --- que es lo que calcula esto--- tiene que perder frente a alguna
   distorsion. Si ganara la cubica, la maquinaria de fases no sirve.
3. Que el grupo espacial de cada estructura RELAJADA se puede medir, que es lo
   unico que responde "en que fase quedo".

Dos fases, porque los entornos estan separados: M3GNet vive en `perovowl-mlff`
y GPAW en `gpaw246`, y no comparten interprete.

Uso:
    # en perovowl-mlff
    python scripts/probar_fases_cspbi3.py --relajar --out build/fases_cspbi3.json
    # en gpaw246
    python scripts/probar_fases_cspbi3.py --gaps --desde build/fases_cspbi3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Potencial interatomico para la relajacion. Entrenado sobre MatPES-PBE, que
#: es el mismo funcional con el que se hacen las etiquetas DFT del pipeline ---
#: relajar con un potencial ajustado a otro funcional meteria un sesgo que
#: despues nadie sabria de donde viene.
MODELO_PES = "M3GNet-PES-MatPES-PBE-2025.2"

#: Lo que ya medimos en esta sesion, con la geometria corregida.
MEDIDO = {
    "a_cubica_A": 6.183,
    "Eg_pbe_eV": 1.097,
    "Eg_pbe_soc_eV": 0.301,
    "Eg_gllbsc_eV": 1.943,
}

#: Literatura. Son ENTRADAS para contrastar, no resultados del calculo.
LITERATURA = {
    "alfa": {"grupo": "Pm-3m", "a_A": 6.18, "Eg_eV": 1.73,
             "nota": "cubica negra; solo estable por encima de ~330 C"},
    "beta": {"grupo": "P4/mbm", "Eg_eV": 1.68, "nota": "tetragonal negra"},
    "gamma": {"grupo": "Pnma", "Eg_eV": 1.75,
              "nota": "ortorrombica negra, octaedros de vertice compartido"},
    "delta": {"grupo": "Pnma", "Eg_eV": 2.82,
              "nota": "amarilla, ARISTA compartida; la estable a 25 C. "
                      "No es una inclinacion de la cubica: otra topologia, "
                      "asi que esta siembra no la genera."},
}


def _padre_cubico():
    """CsPbI3 cubico con la geometria del pipeline."""
    import yaml

    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.structure.build_abx3 import ABX3StructureBuilder

    cfg = yaml.safe_load((ROOT / "config" / "generator.yaml").read_text(encoding="utf-8"))
    gen = HeuristicGenerator(str(ROOT / "config" / "generator.yaml"), random_seed=1)
    cand = gen._make_candidate(
        A_sp=["Cs"], B_sp=["Pb"], X_sp=["I"],
        A_f={"Cs": 1.0}, B_f={"Pb": 1.0}, X_f={"I": 1.0}, mode="pure",
    )
    return ABX3StructureBuilder(cfg).build(cand)


def fase_relajar(salida: Path) -> int:
    """Siembra las fases, las relaja con M3GNet y las ordena por energia."""
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    from buho.structure import fases as F

    import matgl
    from matgl.ext.ase import PESCalculator

    # Nombre de matgl 4.x. El de las versiones 0.x
    # ("M3GNet-MP-2021.2.8-PES") ya no existe, y el fallo que da es un error de
    # autenticacion de HuggingFace que despista: no es que falten permisos, es
    # que el modelo no esta en el registro.
    nombre = MODELO_PES
    modelo = matgl.load_model(nombre)
    calc = PESCalculator(modelo)
    print(f"potencial: {nombre}", flush=True)

    atoms, meta = _padre_cubico()
    print(f"padre cubico: a = {meta['lattice_constant_A']:.4f} A, "
          f"{len(atoms)} atomos", flush=True)

    resultados = []
    for cand in F.generar_candidatas(atoms, b_sites={"Pb"}, x_sites={"I"}):
        est = cand["atoms"].copy()
        n_fu = len(est) // 5           # unidades formula ABX3
        sembrado = F.grupo_espacial(est)

        est.calc = calc
        e0 = float(est.get_potential_energy())
        v0 = float(est.get_volume())
        opt = FIRE(FrechetCellFilter(est), logfile=None)
        convergido = bool(opt.run(fmax=0.05, steps=300))
        e1 = float(est.get_potential_energy())
        v1 = float(est.get_volume())
        est.calc = None
        relajado = F.grupo_espacial(est)

        resultados.append({
            "fase": cand["fase"],
            "glazer": cand["glazer"],
            "grupo_esperado": cand["grupo_esperado"],
            "grupo_sembrado": sembrado.get("simbolo"),
            "grupo_relajado": relajado.get("simbolo"),
            "num_relajado": relajado.get("numero"),
            "n_atomos": len(est),
            "n_formulas": n_fu,
            "convergido": convergido,
            "E_por_formula_eV": round(e1 / n_fu, 5),
            "E_inicial_por_formula_eV": round(e0 / n_fu, 5),
            "volumen_por_formula_A3": round(v1 / n_fu, 4),
            "cambio_volumen_pct": round(100.0 * (v1 - v0) / v0, 2),
            "a_efectivo_A": round(float((v1 / n_fu) ** (1 / 3)), 4),
            "celda": np.array(est.cell).tolist(),
            "posiciones": est.get_positions().tolist(),
            "simbolos": est.get_chemical_symbols(),
        })
        print(f"  {cand['fase']:22s} sembrada={sembrado.get('simbolo'):8s} "
              f"relajada={str(relajado.get('simbolo')):8s} "
              f"E/f.u.={e1 / n_fu:9.4f} eV  dV={100 * (v1 - v0) / v0:+6.2f} %",
              flush=True)

    resultados.sort(key=lambda r: r["E_por_formula_eV"])
    base = resultados[0]["E_por_formula_eV"]
    for r in resultados:
        r["dE_meV_por_formula"] = round(1000 * (r["E_por_formula_eV"] - base), 1)

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(
        {"medido_previamente": MEDIDO, "literatura": LITERATURA,
         "relajacion": f"{MODELO_PES}, FIRE + FrechetCellFilter",
         "fases": resultados}, indent=2), encoding="utf-8")

    print()
    print("=== orden de energia (0 K, M3GNet) ===")
    for r in resultados:
        print(f"  {r['fase']:22s} {r['dE_meV_por_formula']:+8.1f} meV/f.u.  "
              f"{r['grupo_relajado']}")
    ganadora = resultados[0]
    print()
    if ganadora["fase"] == "cubica":
        print("  ATENCION: gana la cubica. CsPbI3 cubico es inestable a 0 K,")
        print("  asi que esto contradice el experimento: revisar la siembra.")
    else:
        print(f"  Gana '{ganadora['fase']}' ({ganadora['grupo_relajado']}), no la cubica.")
        print("  Es lo que dice el experimento: la alfa cubica solo existe >330 C.")
    print(f"\nEscrito {salida}")
    return 0


def fase_gaps(entrada: Path, ecut: int, kpts: int) -> int:
    """Calcula el gap con GLLB-SC de la ganadora y de la cubica."""
    from ase import Atoms
    from gpaw import GPAW, PW, FermiDirac

    datos = json.loads(entrada.read_text(encoding="utf-8"))
    fases = datos["fases"]
    interesantes = [fases[0]]
    cubica = next((f for f in fases if f["fase"] == "cubica"), None)
    if cubica is not None and cubica is not fases[0]:
        interesantes.append(cubica)

    salida = []
    for f in interesantes:
        atoms = Atoms(symbols=f["simbolos"], positions=f["posiciones"],
                      cell=f["celda"], pbc=True)
        calc = GPAW(mode=PW(ecut), xc="GLLBSC",
                    kpts={"size": [kpts] * 3, "gamma": True},
                    occupations=FermiDirac(0.01),
                    convergence={"density": 1e-3, "eigenstates": 1e-4,
                                 "energy": 1e-4},
                    symmetry="off", txt=f"fases_gap_{f['fase']}.txt")
        atoms.calc = calc
        atoms.get_potential_energy()
        homo, lumo = calc.get_homo_lumo()
        resp = calc.hamiltonian.xc.response
        pot = resp.calculate_discontinuity_potential(homo, lumo)
        ks, dxc = resp.calculate_discontinuity(pot)
        salida.append({"fase": f["fase"], "grupo": f["grupo_relajado"],
                       "Eg_ks_eV": round(ks, 4), "Eg_dxc_eV": round(dxc, 4),
                       "Eg_eV": round(ks + dxc, 4)})
        print(f"  {f['fase']:22s} {f['grupo_relajado']:8s} "
              f"KS={ks:6.3f} + Dxc={dxc:6.3f} = {ks + dxc:6.3f} eV", flush=True)

    datos["gaps_gllbsc"] = salida
    entrada.write_text(json.dumps(datos, indent=2), encoding="utf-8")
    print(f"\nActualizado {entrada}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--relajar", action="store_true")
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--desde", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "fases_cspbi3.json")
    ap.add_argument("--ecut", type=int, default=300)
    ap.add_argument("--kpts", type=int, default=2)
    args = ap.parse_args()

    if args.relajar:
        return fase_relajar(args.out)
    if args.gaps:
        if not args.desde:
            ap.error("--gaps necesita --desde")
        return fase_gaps(args.desde, args.ecut, args.kpts)
    ap.error("hace falta --relajar o --gaps")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
