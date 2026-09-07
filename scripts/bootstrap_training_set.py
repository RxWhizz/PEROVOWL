#!/usr/bin/env python3
"""Genera etiquetas de bandgap en la escala actual, para arrancar el surrogate.

Por que hace falta
------------------
El surrogate publicado se entreno con etiquetas de otra escala: gaps de PBE
sobre la geometria vieja (contraccion B-X calibrada solo con yoduros) y sin el
SOC aplicado, porque la tabla no cargaba. CsSnI3 figuraba a 1.075 eV; con la
geometria corregida y SOC son 0.128 eV.

Un modelo entrenado asi no se puede corregir de escala: sumarle el
desplazamiento calibrado lo lleva a un rango arbitrario. Hace falta volver a
etiquetar.

Metodo
------
Composiciones puras ABX3 sobre el espacio quimico del proyecto. Puras a
proposito: son celdas de 5 atomos y cuestan minutos, mientras que una mezcla
necesita la supercelda 2x2x2 de 40 atomos. Con las puras se ancla la escala en
todo el espacio; las mezclas las va aportando el propio bucle al correr.

El objetivo es `Eg_target_eV = Eg_PBE + chi_SOC`, la misma definicion que usa
`discovery.engine`, para que las etiquetas de este arranque y las que produzcan
las rondas siguientes vivan en la misma escala.

Dos fases
---------
El entorno de GPAW no tiene pandas, y no es sitio para instalarlo: es el que
corre los calculos de produccion. Asi que el DFT se hace aparte y la tabla se
arma despues, donde si esta el resto de la pila.

    # dentro de WSL, con el python de gpaw246:
    python scripts/bootstrap_training_set.py --solo-dft --out build/bootstrap_gaps.json

    # desde el entorno del proyecto:
    python scripts/bootstrap_training_set.py --desde build/bootstrap_gaps.json \\
        --out data/discovery/surrogate_training_escala_nueva.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Sitios A inorganicos. MA y FA se sustituyen por Cs al construir la
#: estructura, asi que etiquetarlos aparte duplicaria el mismo calculo con otro
#: nombre -- exactamente el fallo que se corrigio en el generador.
A_SITES = ["Cs", "Rb", "K"]
B_SITES = ["Pb", "Sn", "Ge"]
X_SITES = ["I", "Br", "Cl"]

#: Radios ionicos: A en coordinacion 12, B y X en 6. Se repiten aqui, y no se
#: importan de `ml_surrogate`, porque ese paquete arrastra pandas y esta fase
#: corre en el entorno de GPAW, que no lo tiene.
RADII = {"Cs": 1.88, "Rb": 1.72, "K": 1.64,
         "Pb": 1.19, "Sn": 1.18, "Ge": 0.73,
         "I": 2.20, "Br": 1.96, "Cl": 1.81}

#: La misma contraccion del enlace B-X que usa el constructor de estructuras.
BOND_CONTRACTION = {
    "Pb": {"I": 0.912, "Br": 0.932, "Cl": 0.934},
    "Sn": {"I": 0.920, "Br": 0.924, "Cl": 0.931},
}

ECUT = 300
KPTS = [4, 4, 4]
SMEARING = 0.01


def _celda(a_site: str, b_site: str, x_site: str):
    from ase import Atoms

    a = 2.0 * (RADII[b_site] + RADII[x_site]) * \
        BOND_CONTRACTION.get(b_site, {}).get(x_site, 1.0)
    return Atoms(
        symbols=[a_site, b_site, x_site, x_site, x_site],
        scaled_positions=[
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    ), a


def _gaps(atoms, etiqueta: str) -> tuple[float, float]:
    """(Eg_PBE, Eg_PBE+SOC) de una estructura."""
    from gpaw import GPAW, PW, FermiDirac
    from gpaw.spinorbit import soc_eigenstates

    calc = GPAW(
        mode=PW(ECUT), xc="PBE",
        kpts={"size": KPTS, "gamma": True},
        occupations=FermiDirac(SMEARING),
        convergence={"density": 1e-3, "eigenstates": 1e-4, "energy": 1e-4},
        symmetry="off",
        txt=f"bootstrap_{etiqueta}.txt",
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    n_e = int(round(calc.get_number_of_electrons()))
    nk = len(calc.get_ibz_k_points())

    def gap(eigs, n_ocup):
        return float(min(e[n_ocup] for e in eigs) - max(e[n_ocup - 1] for e in eigs))

    return (gap([calc.get_eigenvalues(kpt=k) for k in range(nk)], n_e // 2),
            gap(soc_eigenstates(calc).eigenvalues(), n_e))


def fase_dft(salida: Path, limite: int = 0) -> int:
    """Calcula los gaps y los vuelca crudos. Sin pandas ni el resto de la pila."""
    combos = [(a, b, x) for a in A_SITES for b in B_SITES for x in X_SITES]
    if limite:
        combos = combos[:limite]

    resultados = []
    for i, (a, b, x) in enumerate(combos, 1):
        atoms, a_lat = _celda(a, b, x)
        etiqueta = f"{a}{b}{x}3"
        try:
            eg_pbe, eg_soc = _gaps(atoms, etiqueta)
        except Exception as exc:  # noqa: BLE001 - un fallo no debe tirar el lote
            print(f"  [{i}/{len(combos)}] {etiqueta}: FALLO "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        resultados.append({"A": a, "B": b, "X": x, "a_lat_A": round(a_lat, 4),
                           "Eg_pbe_eV": round(eg_pbe, 4),
                           "Eg_pbe_soc_eV": round(eg_soc, 4)})
        print(f"  [{i}/{len(combos)}] {etiqueta:9s} "
              f"PBE={eg_pbe:6.3f}  +SOC={eg_soc:6.3f}", flush=True)

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    print(f"\nEscritos {len(resultados)} gaps en {salida}")
    return 0 if resultados else 1


def fase_tabla(entrada: Path, salida: Path) -> int:
    """Arma el CSV de entrenamiento con los descriptores del pipeline."""
    import csv

    sys.path.insert(0, str(ROOT / "src"))
    import yaml

    from buho.generator.heuristic_generator import HeuristicGenerator
    from buho.screening.cascade import ScreeningCascade

    gaps = json.loads(entrada.read_text(encoding="utf-8"))
    gen = HeuristicGenerator(str(ROOT / "config" / "generator.yaml"), random_seed=42)

    filas = []
    for g in gaps:
        a, b, x = g["A"], g["B"], g["X"]
        cand = gen._make_candidate(
            A_sp=[a], B_sp=[b], X_sp=[x],
            A_f={a: 1.0}, B_f={b: 1.0}, X_f={x: 1.0}, mode="pure",
        )
        if cand is None:
            continue
        fila = dict(ScreeningCascade._features(cand))
        fila.update({
            "candidate_id": cand.candidate_id,
            "formula": cand.formula,
            "a_lat_A": g["a_lat_A"],
            "band_gap_gga_eV": g["Eg_pbe_eV"],
            "chi_soc_eV": round(g["Eg_pbe_soc_eV"] - g["Eg_pbe_eV"], 5),
            # La misma definicion que usa discovery.engine para sus etiquetas.
            "Eg_target_eV": g["Eg_pbe_soc_eV"],
            "split": "train",
            "source": "bootstrap_escala_pbe_soc_geom_bx",
            "added_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        filas.append(fila)

    if not filas:
        print("Sin filas: nada que escribir.", file=sys.stderr)
        return 1

    columnas = sorted({k for f in filas for k in f})
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    print(f"Escritas {len(filas)} filas en {salida}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solo-dft", action="store_true",
                    help="Solo calcular los gaps (entorno de GPAW).")
    ap.add_argument("--desde", type=Path, help="JSON de gaps de la fase anterior.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limite", type=int, default=0)
    args = ap.parse_args()

    if args.solo_dft:
        return fase_dft(args.out, args.limite)
    if not args.desde:
        ap.error("hace falta --solo-dft o --desde")
    return fase_tabla(args.desde, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
