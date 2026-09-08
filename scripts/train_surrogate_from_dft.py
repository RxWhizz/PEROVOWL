#!/usr/bin/env python3
"""Entrena un surrogate de energía/átomo DFT desde el cribado single-point.

Target = energy_per_atom_eV de status.json (escrito por GPAW 24.6 → fiable; NO
se lee el gpw, que cross-version con el master del venv da valores corruptos).
Features = descriptores de composición 16D (radios efectivos, Goldschmidt, etc.).

Esto cierra el loop de active learning con un target sólido: un predictor rápido
composición→energía DFT que complementa el Eform de MLFF en la cascada.

Uso:
    python scripts/train_surrogate_from_dft.py [--relax-dir runs/relax_basic]
                                               [--out models/surrogate_energy.pkl]
                                               [--min-converged 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from buho.generator.heuristic_generator import HeuristicGenerator
from buho.screening.cascade import ScreeningCascade
from ml_surrogate.features import BASE_FEATURES, build_X
from ml_surrogate.model import SurrogateEnsemble


def _load_candidate_index() -> dict:
    """candidate_id → GeneratedCandidate (de los jsonl de generación)."""
    idx = {}
    for jl in [ROOT / "data" / "raw" / "generated_candidates.jsonl",
               *(ROOT / "data" / "batches").glob("batch_*/candidates.jsonl")]:
        if jl.exists():
            for c in HeuristicGenerator.load_jsonl(jl):
                idx[c.candidate_id] = c
    return idx


def collect_training_rows(relax_dir: Path, cand_idx: dict) -> pd.DataFrame:
    """Construye filas (features + energy_per_atom) de jobs convergidos."""
    rows = []
    for d in sorted(relax_dir.iterdir()):
        if not d.is_dir():
            continue
        sf = d / "status.json"
        if not sf.exists():
            continue
        try:
            s = json.loads(sf.read_text())
        except Exception:
            continue
        if s.get("status") != "converged":
            continue
        epa = s.get("energy_per_atom_eV")
        if epa is None:
            continue
        cid = s.get("candidate_id", d.name)
        c = cand_idx.get(cid)
        if c is None:
            continue
        feat = ScreeningCascade._features(c)
        feat["candidate_id"] = cid
        feat["formula"] = c.formula
        feat["generation_mode"] = c.generation_mode
        feat["energy_per_atom_eV"] = float(epa)
        rows.append(feat)
    return pd.DataFrame(rows)


def robust_outlier_mask(y: np.ndarray, z: float = 5.0) -> np.ndarray:
    """True = conservar (z-robusto MAD ≤ z)."""
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        return np.ones_like(y, dtype=bool)
    return np.abs(y - med) / (1.4826 * mad) <= z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relax-dir", default=str(ROOT / "runs" / "relax_basic"))
    ap.add_argument("--out", default=str(ROOT / "models" / "surrogate_energy.pkl"))
    ap.add_argument("--min-converged", type=int, default=50)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    cand_idx = _load_candidate_index()
    df = collect_training_rows(Path(args.relax_dir), cand_idx)
    print(f"Jobs convergidos con composición + energía: {len(df)}")
    if len(df) < args.min_converged:
        print(f"  < {args.min_converged} → no se entrena todavía.")
        return 1

    # Guard de outliers (geometrías patológicas)
    y_all = df["energy_per_atom_eV"].values.astype(float)
    keep = robust_outlier_mask(y_all)
    n_out = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    print(f"Outliers de energía descartados: {n_out}  → entrenamiento con {len(df)}")

    feat_cols = list(BASE_FEATURES)
    X = build_X(df, feat_cols)
    y = df["energy_per_atom_eV"].values.astype(float)

    # CV
    from sklearn.model_selection import KFold
    maes = []
    kf = KFold(n_splits=min(args.cv_folds, len(df)), shuffle=True, random_state=42)
    for tr, te in kf.split(X):
        m = SurrogateEnsemble().fit(X[tr], y[tr], feat_cols)
        pred = np.array([m.predict_single(x)[0] for x in X[te]])
        maes.append(float(np.mean(np.abs(pred - y[te]))))
    cv_mae = float(np.mean(maes))
    print(f"CV MAE ({len(maes)}-fold): {cv_mae:.4f} eV/átomo  (rango y: "
          f"{y.min():.2f}..{y.max():.2f})")

    # Modelo final sobre todo
    model = SurrogateEnsemble().fit(X, y, feat_cols)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    metrics = {
        "target": "energy_per_atom_eV",
        "n_train": int(len(df)),
        "n_outliers_dropped": n_out,
        "cv_mae_eV_atom": round(cv_mae, 4),
        "y_min": round(float(y.min()), 3),
        "y_max": round(float(y.max()), 3),
        "features": feat_cols,
        # Solo el nombre: la ruta absoluta viaja dentro del binario
        # distribuido y publica la disposicion de discos de quien entreno.
        "source_relax_dir": Path(args.relax_dir).name,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    Path(str(out).replace(".pkl", ".metrics.json")).write_text(json.dumps(metrics, indent=2))
    print(f"Modelo guardado: {out}")
    print(f"Métricas: {Path(str(out).replace('.pkl', '.metrics.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
