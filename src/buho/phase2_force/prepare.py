"""Preparacion de jobs Fase 2A para DFT E+F(+stress)."""

from __future__ import annotations

import argparse
import json
import shutil
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from buho.phase2_force import ROOT
from buho.phase2_force.common import RUNS_DIR, display_path, label_plan_for_formula, read_csv, write_json
from buho.phase2_force.selection import load_candidate_index

# k-points para superceldas: [2,2,2] (test de convergencia 2026-06-09: Γ-only da 0.2 eV/átomo
# de error en Sn; [2,2,2] converge a 0.015 vs [3,3,3]). Con PBE single-point ~6 GB/job.
SUPERCELL_KPTS = [2, 2, 2]
# Single-points por candidato: config 0 = semilla relajada; 1..K-1 = rattled (diversidad).
N_RATTLE = 4
RATTLE_STDEV = 0.08  # Å

# Relajacion previa con MACE-MP-0: la semilla cubica idealizada usa lattice_est = 2√2(r_B+r_X)
# que SOBRE-expande ~50% (CsSnI3 sale a 9.56 Å vs ~6.1 real) -> estructura casi metalica ->
# SCF oscila y los forces son no-fisicos. Relajar con MACE-MP-0 contrae la red al equilibrio
# de MACE; el DFT single-point alrededor de ese equilibrio da el Δ(DFT-MACE) que necesita el
# fine-tuning. Calculadora cacheada (se carga una vez por corrida de prepare).
_MACE_CALC = None


def _get_mace_calc():
    global _MACE_CALC
    if _MACE_CALC is None:
        from mace.calculators import mace_mp
        _MACE_CALC = mace_mp(model="small", dispersion=False,
                             default_dtype="float64", device="cpu")
    return _MACE_CALC


def _relax_with_mace(structure_path: Path, fmax: float = 0.05, steps: int = 120) -> dict:
    """Relaja in-place la estructura con MACE-MP-0 (celda+posiciones). Sobrescribe el cif."""
    from ase.filters import FrechetCellFilter
    from ase.io import read as _read
    from ase.io import write as _write
    from ase.optimize import FIRE
    atoms = _read(str(structure_path))
    vol0 = atoms.get_volume() / len(atoms)
    atoms.calc = _get_mace_calc()
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)
    vol1 = atoms.get_volume() / len(atoms)
    atoms.calc = None
    _write(str(structure_path), atoms)
    return {"mace_relaxed": True, "mace_converged": bool(converged),
            "vol_per_atom_before": round(float(vol0), 2),
            "vol_per_atom_after": round(float(vol1), 2)}


INPUT_TEMPLATE = string.Template(r'''#!/usr/bin/env python3
"""Job Fase 2A: PBE single-point E+F(+stress) sobre K estructuras rattled -> extxyz MACE.

Sin Hubbard U (consistente con MPtrj/MACE-MP-0) y sin FIRE: para datos de fine-tuning de
un MLIP se quieren E+F sobre estructuras DIVERSAS (rattled), no la relajacion al minimo.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from gpaw import GPAW, PW
from gpaw.eigensolvers import Davidson
from gpaw.mixer import Mixer
from gpaw.mpi import world


JOB_ROOT = Path(__file__).resolve().parent
STRUCTURE = JOB_ROOT / "structure.cif"
METADATA = json.loads((JOB_ROOT / "metadata.json").read_text())
LABELS = json.loads(r"""$labels_json""")
N_CORES = $n_cores
KPTS_SUPERCELL = $kpts_supercell
N_RATTLE = $n_rattle
RATTLE_STDEV = $rattle_stdev
IS_MASTER = world.rank == 0


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _read_status():
    base = {
        "candidate_id": METADATA.get("candidate_id"),
        "formula": METADATA.get("formula"),
        "phase2_batch_id": METADATA.get("phase2_batch_id"),
        "selection_rank": METADATA.get("selection_rank"),
        "n_labels_expected": len(LABELS),
        "labels_expected": LABELS,
    }
    p = JOB_ROOT / "status.json"
    if not p.exists():
        return base
    try:
        status = json.loads(p.read_text())
    except Exception:
        return base
    if not status.get("candidate_id"):
        base.update(status)
        return base
    return status


def _write_status(update):
    if not IS_MASTER:
        world.barrier()
        return
    status = _read_status()
    status.update(update)
    tmp = JOB_ROOT / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(JOB_ROOT / "status.json")
    world.barrier()


def _kpts_for_atoms(atoms):
    # [2,2,2] para superceldas (test de convergencia 2026-06-09: Γ-only da 0.2 eV/átomo de
    # error en Sn; [2,2,2] converge a 0.015 eV/átomo vs [3,3,3]).
    return KPTS_SUPERCELL if len(atoms) > 10 else [4, 4, 4]


def _calc_kwargs(atoms):
    kpts = _kpts_for_atoms(atoms)
    has_sn = "Sn" in (METADATA.get("formula") or "")
    # SIN Hubbard U (consistente MPtrj). Sn: smearing ancho (0.2) + mixer estandar (0.05)
    # para suavizar la oscilacion SCF — NO Mixer(0.002) que es lentisimo, ni U.
    return {
        "mode": PW(450),
        "xc": "PBE",
        "kpts": {"size": kpts, "gamma": True},
        "occupations": {"name": "fermi-dirac", "width": 0.2 if has_sn else 0.05},
        "eigensolver": Davidson(niter=3),
        # parallel: omitido -> GPAW auto-decide (robusto para cualquier N_CORES; evita el
        # bug de divisibilidad kpt/cores, p.ej. 11 cores con 4 k-points IBZ).
        "convergence": {"density": 1e-4, "eigenstates": 1e-6, "energy": 1e-5},
        "mixer": Mixer(0.05, 8, 50),
        "maxiter": 333,
        # Nombre legacy r2scan.txt: el runner/monitor detectan progreso SCF y stall por este
        # archivo (label_log_paths usa rel_dir/r2scan.txt). Cada config rattled lo reescribe;
        # el mtime fresco evita falsos "no-SCF stall".
        "txt": str(JOB_ROOT / "pbe" / "r2scan.txt"),
    }


def _json_float_list(array):
    if array is None:
        return None
    return np.asarray(array, dtype=float).tolist()


def _single_point(atoms, calc):
    # UN proceso MPI = UN calculo GPAW. Cualquier intento de correr varios GPAW en
    # secuencia dentro del mismo proceso (calc nuevo por config O calc compartido)
    # produce deadlock MPI en GPAW 24.6 + OpenMPI (collectives desfasados, CPU al
    # 100% sin output). job.sh lanza un mpiexec POR config (aislamiento total).
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    fmax = float(np.linalg.norm(forces, axis=1).max())
    stress = None
    stress_error = None
    try:
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception as exc:
        stress_error = str(exc)
    return energy, forces, fmax, stress, stress_error


def run_config(k: int) -> None:
    """Computa SOLO la config k (proceso MPI fresco). Idempotente via frame_k.json."""
    label = LABELS[0]   # unico label PBE (sin U-scan)
    work = JOB_ROOT / label["relative_dir"]
    work.mkdir(parents=True, exist_ok=True)
    frame_path = work / f"frame_{k}.json"
    if frame_path.exists():
        try:
            if json.loads(frame_path.read_text()).get("status") == "ok":
                return   # ya computada (resume)
        except Exception:
            pass

    if k == 0:
        _write_status({"status": "running", "started_at": _now()})
        if IS_MASTER:
            for old in (work / "label.extxyz", work / "metrics.json"):
                if old.exists():
                    old.unlink()
    world.barrier()

    t0 = time.time()
    atoms = read(str(STRUCTURE))
    if k > 0:
        atoms.rattle(stdev=RATTLE_STDEV, seed=1000 + k)

    try:
        calc = GPAW(**_calc_kwargs(atoms))
        energy, forces, fmax, stress, stress_error = _single_point(atoms, calc)
        # SinglePointCalculator: forma canonica de guardar E+F+stress en extxyz para
        # MLIP (no poner energy/stress en info: colisiona al escribir).
        at_out = atoms.copy()
        # Limpiar claves heredadas del CIF que rompen el parser extxyz de ASE
        # (occupancy contiene JSON con comillas escapadas; spacegroup/unit_cell son
        # metadata estructural irrelevante para el entrenamiento de MACE).
        for _k in ("occupancy", "spacegroup", "unit_cell"):
            at_out.info.pop(_k, None)
        spc = {"energy": energy, "forces": forces}
        if stress is not None:
            spc["stress"] = stress
        at_out.calc = SinglePointCalculator(at_out, **spc)
        at_out.info.update({
            "candidate_id": METADATA["candidate_id"],
            "formula": METADATA.get("formula"),
            "config_index": k,
            "rattle_stdev": (0.0 if k == 0 else RATTLE_STDEV),
            "method": "PBE",
            "fidelity": "phase2_force",
        })
        frame = {
            "config_index": k, "status": "ok",
            "energy_eV": energy, "energy_per_atom_eV": energy / len(atoms),
            "forces_max_eVA": fmax, "stress": _json_float_list(stress),
            "stress_available": stress is not None, "stress_error": stress_error,
            "n_atoms": len(atoms), "kpts": _kpts_for_atoms(atoms),
            "elapsed_s": round(time.time() - t0, 1), "finished_at": _now(),
        }
        if IS_MASTER:
            write(str(work / "label.extxyz"), at_out, format="extxyz", append=True)
            frame_path.write_text(json.dumps(frame, indent=2))
    except Exception as exc:
        if IS_MASTER:
            frame_path.write_text(json.dumps({
                "config_index": k, "status": "failed", "error_message": str(exc),
                "elapsed_s": round(time.time() - t0, 1), "finished_at": _now(),
            }, indent=2))
            (work / f"error_config{k}.txt").write_text(traceback.format_exc())
    world.barrier()


def finalize() -> None:
    """Agrega frame_*.json -> metrics.json + status final. Correr SIN mpiexec."""
    label = LABELS[0]
    work = JOB_ROOT / label["relative_dir"]
    frames = []
    for k in range(N_RATTLE):
        p = work / f"frame_{k}.json"
        if p.exists():
            try:
                frames.append(json.loads(p.read_text()))
                continue
            except Exception:
                pass
        frames.append({"config_index": k, "status": "failed",
                       "error_message": "frame_json_missing (proceso murio sin escribir)"})
    n_ok = sum(1 for m in frames if m.get("status") == "ok")
    status = "converged" if n_ok == N_RATTLE else ("partial" if n_ok > 0 else "failed")
    metrics = {
        "status": status,
        "candidate_id": METADATA["candidate_id"],
        "formula": METADATA.get("formula"),
        "method": "PBE",
        "n_frames": n_ok,
        "n_frames_requested": N_RATTLE,
        "rattle_stdev": RATTLE_STDEV,
        "xc": "PBE",
        "frames": frames,
        "finished_at": _now(),
    }
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_status({"status": status, "n_frames": n_ok,
                   "n_frames_requested": N_RATTLE, "finished_at": _now()})
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    elif args.config_index is not None:
        run_config(args.config_index)
    else:
        raise SystemExit("Usa --config-index K (bajo mpiexec) o --finalize (sin mpiexec).")
''')


JOB_SH_TEMPLATE = string.Template(r'''#!/bin/bash
# Un mpiexec POR config (aislamiento de procesos MPI; ver input.py).
# El runner exporta GPAW_SETUP_PATH y NCORES; conda env ya activo.
set -u
cd "$$(dirname "$$0")"
for k in $config_indices; do
    mpiexec -n "$${NCORES:-$n_cores}" python input.py --config-index "$$k"
done
exec python input.py --finalize
''')


def _batch_csv(batch_id: int) -> Path:
    return ROOT / "data" / "mace_finetune" / "batches" / f"batch_{batch_id:03d}.csv"


def _job_dir(batch_id: int, candidate_id: str, runs_dir: Path = RUNS_DIR) -> Path:
    return runs_dir / f"batch_{batch_id:03d}" / candidate_id


def _load_candidate(candidate_id: str, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    data = dict(index[candidate_id])
    data.pop("_json_source", None)
    return data


def _build_phase1_seed_structure(candidate: dict[str, Any], cfg: dict[str, Any], out_dir: Path) -> None:
    """Construye la misma semilla cubica idealizada de Fase 1 sin depender de pandas."""
    import numpy as np
    from ase import Atoms
    from ase.build import make_supercell
    from ase.io.trajectory import Trajectory

    structure_cfg = cfg.get("structure", {})
    organic_placeholder = structure_cfg.get("organic_A_placeholder", "Cs")
    supercell_mixed = list(structure_cfg.get("supercell_mixed", [2, 2, 2]))
    supercell_pure = list(structure_cfg.get("supercell_pure", [1, 1, 1]))
    formats = list(structure_cfg.get("export_formats", ["cif", "poscar", "traj"]))

    # Corrección de red: a0_est_A/lattice_constant_A vienen de lattice_est = 2√2·(r_B+r_X),
    # que sobre-expande √2× (CsSnI3 → 9.56 Å). La red cúbica de perovskita Pm-3m es
    # a = 2·(r_B+r_X) = lattice_est/√2 (B-X = a/2). Esto arranca el MACE relax cerca del
    # equilibrio (rápido) en vez de desde una estructura casi metálica sobre-expandida.
    a0 = float(candidate.get("a0_est_A") or candidate.get("lattice_constant_A") or 8.49) / (2.0 ** 0.5)
    a_species = list(candidate.get("A_site_species", []))
    b_species = list(candidate.get("B_site_species", []))
    x_species = list(candidate.get("X_site_species", []))
    fractions = candidate.get("fractions", {})
    is_organic = bool(candidate.get("is_organic_A"))
    is_mixed = len(a_species) > 1 or len(b_species) > 1 or len(x_species) > 1

    a_struct = organic_placeholder if is_organic else a_species[0]
    b_struct = b_species[0]
    x_struct = x_species[0]
    scaled = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]
    atoms = Atoms(
        symbols=[a_struct, b_struct, x_struct, x_struct, x_struct],
        scaled_positions=scaled,
        cell=[a0, a0, a0],
        pbc=True,
    )
    sc = supercell_mixed if is_mixed else supercell_pure
    if sc != [1, 1, 1]:
        atoms = make_supercell(atoms, np.diag(sc))

    if is_mixed:
        rng = np.random.RandomState(int(cfg.get("random_seed", 42)))
        symbols = list(atoms.get_chemical_symbols())

        def _ase_symbol(species: str) -> str:
            return organic_placeholder if species in {"MA", "FA"} else species

        def _assign(site_symbols: list[str], species_list: list[str], fracs: dict[str, float]) -> None:
            if len(species_list) <= 1:
                return
            idxs = [idx for idx, sym in enumerate(symbols) if sym in site_symbols]
            total = len(idxs)
            shuffled = idxs.copy()
            rng.shuffle(shuffled)
            remaining = total
            offset = 0
            for species in species_list[:-1]:
                n = min(remaining, int(round(float(fracs.get(species, 0.0)) * total)))
                for idx in shuffled[offset:offset + n]:
                    symbols[idx] = _ase_symbol(species)
                offset += n
                remaining -= n
            for idx in shuffled[offset:]:
                symbols[idx] = _ase_symbol(species_list[-1])

        _assign([a_struct, organic_placeholder], a_species, fractions.get("A", {}))
        _assign(b_species, b_species, fractions.get("B", {}))
        _assign(x_species, x_species, fractions.get("X", {}))
        atoms.set_chemical_symbols(symbols)

    out_dir.mkdir(parents=True, exist_ok=True)
    if "cif" in formats:
        atoms.write(str(out_dir / "structure.cif"))
    if "poscar" in formats:
        atoms.write(str(out_dir / "POSCAR"), format="vasp")
    if "traj" in formats:
        with Trajectory(str(out_dir / "structure.traj"), "w") as traj:
            traj.write(atoms)

    metadata = {
        "candidate_id": candidate["candidate_id"],
        "formula": candidate["formula"],
        "reduced_formula": candidate.get("reduced_formula"),
        "generation_mode": candidate.get("generation_mode"),
        "A_site_species": a_species,
        "B_site_species": b_species,
        "X_site_species": x_species,
        "fractions": fractions,
        "molecular_A_placeholder": is_organic,
        "lattice_constant_A": round(a0, 4),
        "supercell": sc,
        "n_atoms": len(atoms),
        "tolerance_t": candidate.get("tolerance_t"),
        "oct_factor": candidate.get("oct_factor"),
        "random_seed": cfg.get("random_seed", 42),
        "build_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_input(job_dir: Path, labels: list[dict[str, Any]], n_cores: int) -> None:
    script = INPUT_TEMPLATE.substitute(
        labels_json=json.dumps(labels, indent=2),
        n_cores=int(n_cores),
        kpts_supercell=json.dumps(SUPERCELL_KPTS),
        n_rattle=int(N_RATTLE),
        rattle_stdev=repr(float(RATTLE_STDEV)),
    )
    path = job_dir / "input.py"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    job_sh = JOB_SH_TEMPLATE.substitute(
        config_indices=" ".join(str(k) for k in range(N_RATTLE)),
        n_cores=int(n_cores),
    )
    sh_path = job_dir / "job.sh"
    sh_path.write_text(job_sh, encoding="utf-8")
    sh_path.chmod(0o755)


def _write_job_metadata(job_dir: Path, row: dict[str, str], labels: list[dict[str, Any]],
                        mace_info: dict[str, Any] | None = None) -> None:
    metadata_path = job_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update({
        "phase": "2A",
        "fidelity": "phase2_force",
        "phase2_batch_id": int(row["phase2_batch_id"]),
        "selection_rank": int(row["selection_rank"]),
        "slot_in_batch": int(row["slot_in_batch"]),
        "dft_labels_expected": labels,
        "n_rattle": N_RATTLE,
        "rattle_stdev": RATTLE_STDEV,
        "kpts_supercell": SUPERCELL_KPTS,
        "mace_prerelax": mace_info or {},
        "dft_policy": "MACE-MP-0 relax -> PBE single-point E+F sobre K rattled (sin U, MPtrj)",
        "selection_row": row,
        "prepared_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_status(job_dir: Path, row: dict[str, str], labels: list[dict[str, Any]]) -> None:
    status = {
        "status": "pending",
        "candidate_id": row["candidate_id"],
        "formula": row["formula"],
        "phase2_batch_id": int(row["phase2_batch_id"]),
        "selection_rank": int(row["selection_rank"]),
        "n_labels_expected": len(labels),
        "labels_expected": labels,
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (job_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_batch(batch_id: int, config_path: Path = ROOT / "config" / "generator.yaml",
                  runs_dir: Path = RUNS_DIR, n_cores: int = 8, limit: int | None = None,
                  dry_run: bool = False) -> dict[str, Any]:
    batch_csv = _batch_csv(batch_id)
    if not batch_csv.exists():
        raise FileNotFoundError(f"No existe {batch_csv}; corre primero phase2_force_select.")

    rows = read_csv(batch_csv)
    if limit is not None:
        rows = rows[:limit]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate_index = load_candidate_index(ROOT)
    prepared: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    planned: list[dict[str, Any]] = []

    for row in rows:
        cid = row["candidate_id"]
        labels = label_plan_for_formula(row["formula"])
        job_dir = _job_dir(batch_id, cid, runs_dir)
        planned.append({
            "candidate_id": cid,
            "formula": row["formula"],
            "job_dir": display_path(job_dir),
            "labels": labels,
        })
        if dry_run:
            continue
        if cid not in candidate_index:
            missing.append(cid)
            continue

        st_path = job_dir / "status.json"
        if st_path.exists():
            try:
                st = json.loads(st_path.read_text(encoding="utf-8"))
                if st.get("status") in {"running", "converged", "partial", "failed"}:
                    skipped.append(cid)
                    continue
            except Exception:
                pass

        candidate = _load_candidate(cid, candidate_index)
        _build_phase1_seed_structure(candidate, cfg, job_dir)
        # Relajar la semilla con MACE-MP-0 (corrige la sobre-expansion de lattice_est y da
        # estructuras fisicas -> SCF converge y los forces DFT son significativos).
        try:
            mace_info = _relax_with_mace(job_dir / "structure.cif")
        except Exception as exc:
            mace_info = {"mace_relaxed": False, "mace_error": str(exc)[:200]}
        for label in labels:
            (job_dir / label["relative_dir"]).mkdir(parents=True, exist_ok=True)
        _write_input(job_dir, labels, n_cores=n_cores)
        _write_job_metadata(job_dir, row, labels, mace_info)
        _write_status(job_dir, row, labels)
        if config_path.exists():
            shutil.copy2(config_path, job_dir / "generator_config.yaml")
        prepared.append(cid)

    manifest = {
        "batch_id": batch_id,
        "batch_csv": str(batch_csv.relative_to(ROOT)),
        "runs_dir": display_path(runs_dir / f"batch_{batch_id:03d}"),
        "n_candidates": len(rows),
        "n_planned": len(planned),
        "n_prepared": len(prepared),
        "n_skipped": len(skipped),
        "n_missing_candidate_json": len(missing),
        "dry_run": dry_run,
        "planned_examples": planned[:10],
        "prepared_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if not dry_run:
        write_json(runs_dir / f"batch_{batch_id:03d}" / "phase2_force_batch_manifest.json", manifest)
    return manifest


def prepare_all(config_path: Path = ROOT / "config" / "generator.yaml", runs_dir: Path = RUNS_DIR,
                n_cores: int = 8, dry_run: bool = False) -> dict[str, Any]:
    batch_files = sorted((ROOT / "data" / "mace_finetune" / "batches").glob("batch_*.csv"))
    manifests = []
    for path in batch_files:
        batch_id = int(path.stem.split("_")[1])
        manifests.append(prepare_batch(batch_id, config_path, runs_dir, n_cores, dry_run=dry_run))
    summary = {
        "n_batches": len(manifests),
        "n_candidates": sum(m["n_candidates"] for m in manifests),
        "n_prepared": sum(m["n_prepared"] for m in manifests),
        "dry_run": dry_run,
        "batches": manifests,
    }
    if not dry_run:
        write_json(runs_dir / "phase2_force_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara jobs DFT Fase 2A.")
    parser.add_argument("--batch-id", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "config" / "generator.yaml"))
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.all and args.batch_id is None:
        raise SystemExit("Usa --batch-id N o --all")
    if args.all:
        result = prepare_all(Path(args.config), Path(args.runs_dir), args.cores, dry_run=args.dry_run)
    else:
        result = prepare_batch(args.batch_id, Path(args.config), Path(args.runs_dir),
                               args.cores, args.limit, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
