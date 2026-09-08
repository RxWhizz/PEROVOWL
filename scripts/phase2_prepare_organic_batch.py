#!/usr/bin/env python3
"""Prepara batch_010 de Fase 2A con moléculas MA/FA EXPLÍCITAS (no placeholder Cs).

Los batches 0-9 usan el placeholder orgánico (MA/FA→Cs), así que el dataset MACE no
contiene hidrógenos/orgánicos. Este batch construye superceldas 2×2×2 con las moléculas
reales (MA=CH3NH3, FA=CH(NH2)2) en los sitios A, orientación aleatoria reproducible,
MACE-MP-0 prerelax, y el MISMO input.py del pipeline (PBE single-point E+F+stress × 4
rattled, [2,2,2] k-points).

Selección: 50 candidatos orgánicos de phase2_candidates_1000.csv, deduplicados por
composición CUANTIZADA (8 sitios A / 8 B / 24 X) y diversificados por familia B.

Uso: PYTHONPATH=src .venv/bin/python3 scripts/phase2_prepare_organic_batch.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ase import Atoms
from ase.build import make_supercell

from buho.phase2_force.prepare import (INPUT_TEMPLATE, N_RATTLE, RATTLE_STDEV,
                                       SUPERCELL_KPTS)
from buho.phase2_force.common import label_plan_for_formula
from buho.phase2_force.selection import load_candidate_index

BATCH_ID = 10
N_JOBS = 50
RUNS_DIR = ROOT / "local_runs" / "phase2_force"
SEED = 4242

# ── Geometrías moleculares (guess razonable; MACE prerelax las refina) ──────────────
def _ma_atoms() -> Atoms:
    """CH3NH3+ : C-N 1.50 Å en z, H tetraédricos."""
    pos = [("C", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.50))]
    for phi in (90.0, 210.0, 330.0):     # H del C, apuntando lejos del N
        r = np.deg2rad(phi)
        pos.append(("H", (1.09 * 0.943 * np.cos(r), 1.09 * 0.943 * np.sin(r), -1.09 * 0.334)))
    for phi in (30.0, 150.0, 270.0):     # H del N, escalonados
        r = np.deg2rad(phi)
        pos.append(("H", (1.04 * 0.940 * np.cos(r), 1.04 * 0.940 * np.sin(r), 1.50 + 1.04 * 0.342)))
    return Atoms([s for s, _ in pos], positions=[p for _, p in pos])


def _fa_atoms() -> Atoms:
    """CH(NH2)2+ : plano, C central, N-C-N ~124°."""
    pos = [("C", (0.0, 0.0, 0.0)), ("H", (-1.09, 0.0, 0.0))]
    for sgn in (1.0, -1.0):
        n = np.array([1.32 * np.cos(np.deg2rad(62)), sgn * 1.32 * np.sin(np.deg2rad(62)), 0.0])
        pos.append(("N", tuple(n)))
        for dphi in (-60.0, 60.0):
            a = np.deg2rad(sgn * 62 + sgn * dphi)
            pos.append(("H", tuple(n + 1.01 * np.array([np.cos(a), np.sin(a), 0.0]))))
    return Atoms([s for s, _ in pos], positions=[p for _, p in pos])


MOLECULES = {"MA": _ma_atoms(), "FA": _fa_atoms()}


def _random_rotation(rng: np.random.RandomState) -> np.ndarray:
    """Matriz de rotación uniforme (QR de gaussiana)."""
    m = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(m)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# ── Cuantización de fracciones a sitios discretos ───────────────────────────────────
def _quantize(species: list[str], fracs: dict[str, float], n_sites: int) -> list[str]:
    """Reparte n_sites entre especies según fracciones (residuo mayor)."""
    if len(species) == 1:
        return [species[0]] * n_sites
    raw = {s: float(fracs.get(s, 0.0)) * n_sites for s in species}
    base = {s: int(np.floor(v)) for s, v in raw.items()}
    left = n_sites - sum(base.values())
    for s in sorted(species, key=lambda s: raw[s] - base[s], reverse=True)[:left]:
        base[s] += 1
    out = []
    for s in species:
        out.extend([s] * base[s])
    return out


def _composition_key(a: list[str], b: list[str], x: list[str]) -> str:
    return json.dumps([sorted(Counter(a).items()), sorted(Counter(b).items()),
                       sorted(Counter(x).items())])


# ── Construcción de la supercelda con moléculas reales ──────────────────────────────
def build_structure(candidate: dict, rng: np.random.RandomState) -> tuple[Atoms, dict]:
    a_species = list(candidate.get("A_site_species", []))
    b_species = list(candidate.get("B_site_species", []))
    x_species = list(candidate.get("X_site_species", []))
    fracs = candidate.get("fractions", {})

    a0_est = float(candidate.get("a0_est_A") or candidate.get("lattice_constant_A") or 8.49)
    a0 = a0_est / np.sqrt(2.0)           # misma corrección √2 que el pipeline
    if any(s in MOLECULES for s in a_species):
        a0 *= 1.04                        # MA/FA expanden la red un poco; MACE ajusta

    a_assign = _quantize(a_species, fracs.get("A", {}), 8)
    b_assign = _quantize(b_species, fracs.get("B", {}), 8)
    x_assign = _quantize(x_species, fracs.get("X", {}), 24)
    rng.shuffle(a_assign)
    rng.shuffle(b_assign)
    rng.shuffle(x_assign)

    cell = np.eye(3) * (2 * a0)
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    n_mol = Counter()
    ai = bi = xi = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                org = np.array([i, j, k], dtype=float) * a0
                a_sp = a_assign[ai]; ai += 1
                site = org
                if a_sp in MOLECULES:
                    mol = MOLECULES[a_sp].copy()
                    p = mol.get_positions()
                    p -= p.mean(axis=0)
                    p = p @ _random_rotation(rng).T
                    for s, pp in zip(mol.get_chemical_symbols(), p):
                        symbols.append(s)
                        positions.append(site + pp)
                    n_mol[a_sp] += 1
                else:
                    symbols.append(a_sp)
                    positions.append(site)
                b_sp = b_assign[bi]; bi += 1
                symbols.append(b_sp)
                positions.append(org + 0.5 * a0)
                for off in ((0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)):
                    symbols.append(x_assign[xi]); xi += 1
                    positions.append(org + np.array(off) * a0)

    atoms = Atoms(symbols, positions=np.array(positions), cell=cell, pbc=True)
    atoms.wrap()
    info = {"n_molecules": dict(n_mol), "a0_init_A": round(float(a0), 4),
            "a_assign": dict(Counter(a_assign)), "b_assign": dict(Counter(b_assign)),
            "x_assign": dict(Counter(x_assign))}
    return atoms, info


def relax_with_mace(atoms: Atoms, fmax: float = 0.07, steps: int = 150) -> dict:
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from mace.calculators import mace_mp
    if not hasattr(relax_with_mace, "_calc"):
        relax_with_mace._calc = mace_mp(model="small", dispersion=False,
                                        default_dtype="float32", device="cpu")
    vol0 = atoms.get_volume() / len(atoms)
    atoms.calc = relax_with_mace._calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)
    fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    e_pa = float(atoms.get_potential_energy()) / len(atoms)
    atoms.calc = None
    return {"mace_relaxed": True, "mace_converged": bool(converged),
            "mace_fmax_final": round(fmax_final, 4),
            "mace_energy_per_atom": round(e_pa, 4),
            "vol_per_atom_before": round(float(vol0), 2),
            "vol_per_atom_after": round(float(atoms.get_volume() / len(atoms)), 2)}


# ── Selección de candidatos ──────────────────────────────────────────────────────────
def select_candidates(index: dict) -> list[dict]:
    rows = list(csv.DictReader(open(ROOT / "data/mace_finetune/phase2_candidates_1000.csv")))
    rng = np.random.RandomState(SEED)
    seen_keys: set[str] = set()
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:                       # ya vienen ordenados por selection_rank
        cid = row["candidate_id"]
        if cid not in index:
            continue
        cand = index[cid]
        a_sp = list(cand.get("A_site_species", []))
        if not any(s in MOLECULES for s in a_sp):
            continue
        fr_a = cand.get("fractions", {}).get("A", {})
        a_assign = _quantize(a_sp, fr_a, 8)
        if not any(s in MOLECULES for s in a_assign):
            continue                       # la fracción orgánica cuantiza a 0 moléculas
        key = _composition_key(a_assign,
                               _quantize(list(cand.get("B_site_species", [])),
                                         cand.get("fractions", {}).get("B", {}), 8),
                               _quantize(list(cand.get("X_site_species", [])),
                                         cand.get("fractions", {}).get("X", {}), 24))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        by_family[row.get("b_family", "?")].append(row)

    # round-robin por familia B para diversidad
    selected: list[dict] = []
    fams = sorted(by_family, key=lambda f: -len(by_family[f]))
    i = 0
    while len(selected) < N_JOBS and any(by_family[f] for f in fams):
        f = fams[i % len(fams)]
        if by_family[f]:
            selected.append(by_family[f].pop(0))
        i += 1
    return selected[:N_JOBS]


def main() -> None:
    index = load_candidate_index(ROOT)
    selected = select_candidates(index)
    print(f"seleccionados: {len(selected)} | familias: "
          f"{dict(Counter(r.get('b_family') for r in selected))}", flush=True)

    batch_dir = RUNS_DIR / f"batch_{BATCH_ID:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for slot, row in enumerate(selected):
        cid = row["candidate_id"]
        cand = dict(index[cid])
        job_dir = batch_dir / cid
        if (job_dir / "status.json").exists():
            print(f"[{slot:2d}] {cid} ya existe — skip", flush=True)
            continue
        rng = np.random.RandomState(SEED + slot)
        atoms, build_info = build_structure(cand, rng)
        job_dir.mkdir(parents=True, exist_ok=True)
        atoms.write(str(job_dir / "structure.cif"))
        t0 = datetime.now(timezone.utc)
        try:
            mace_info = relax_with_mace(atoms)
            atoms.write(str(job_dir / "structure.cif"))
        except Exception as exc:
            mace_info = {"mace_relaxed": False, "mace_error": str(exc)[:200]}
        labels = label_plan_for_formula(row["formula"])

        script = INPUT_TEMPLATE.substitute(
            labels_json=json.dumps(labels, indent=2), n_cores=8,
            kpts_supercell=json.dumps(SUPERCELL_KPTS), n_rattle=int(N_RATTLE),
            rattle_stdev=repr(float(RATTLE_STDEV)))
        (job_dir / "input.py").write_text(script)
        (job_dir / "input.py").chmod(0o755)

        metadata = {
            "candidate_id": cid, "formula": row["formula"],
            "phase": "2A", "fidelity": "phase2_force",
            "phase2_batch_id": BATCH_ID, "selection_rank": int(row["selection_rank"]),
            "slot_in_batch": slot, "dft_labels_expected": labels,
            "n_rattle": N_RATTLE, "rattle_stdev": RATTLE_STDEV,
            "kpts_supercell": SUPERCELL_KPTS,
            "organic_real": True, "n_atoms": len(atoms),
            "build": build_info, "mace_prerelax": mace_info,
            "dft_policy": "MA/FA explicitos + MACE relax -> PBE single-point E+F (sin U)",
            "prepared_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        (job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (job_dir / "status.json").write_text(json.dumps({
            "status": "pending", "candidate_id": cid, "formula": row["formula"],
            "phase2_batch_id": BATCH_ID, "selection_rank": int(row["selection_rank"]),
            "n_labels_expected": len(labels), "labels_expected": labels,
            "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }, indent=2))

        out_row = dict(row)
        out_row.update({"phase2_batch_id": BATCH_ID, "slot_in_batch": slot,
                        "n_dft_labels_expected": 1, "dft_plan": "pbe_organic_real"})
        csv_rows.append(out_row)
        dt = (datetime.now(timezone.utc) - t0).total_seconds()
        print(f"[{slot:2d}] {row['formula']:34s} n_at={len(atoms):3d} "
              f"mol={build_info['n_molecules']} vol/at={mace_info.get('vol_per_atom_after','?')} "
              f"fmax={mace_info.get('mace_fmax_final','?')} ({dt:.0f}s)", flush=True)

    if csv_rows:
        out_csv = ROOT / "data/mace_finetune/batches" / f"batch_{BATCH_ID:03d}.csv"
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nCSV: {out_csv}")
    manifest = {"batch_id": BATCH_ID, "n_prepared": len(csv_rows),
                "organic_real": True, "prepared_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    (batch_dir / "phase2_force_batch_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {batch_dir}/phase2_force_batch_manifest.json")


if __name__ == "__main__":
    main()
