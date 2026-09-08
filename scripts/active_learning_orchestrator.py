#!/usr/bin/env python3
"""Orquestador autónomo del loop de active learning (Fase 1, composicional).

Encadena batches con reentrenamiento + evaluación + criterio de paro:

  advance():
    1. ¿el DFT del batch actual terminó? (pending==0 en su relax dir)
    2. finalizar: acumular (features + energy_per_atom) de jobs convergidos
    3. reentrenar surrogate de energía con TRAIN/TEST split fijo (held-out 15%
       por hash de candidate_id → estable entre batches) → train_mae, test_mae,
       overfit_ratio = test/train. Loguear curva de aprendizaje.
    4. ¿parar? max_batches | convergencia (Δtest_mae pequeño 2 batches) |
       saturación (pocos candidatos nuevos)
    5. si no → generar+cribar+lanzar el siguiente batch

Estado: data/batches/orchestrator_state.json
Curva:  models/surrogate_energy_learning_curve.csv
Datos acumulados: data/processed/dft_accumulated.csv

Uso:
    python scripts/active_learning_orchestrator.py advance   # un paso
    python scripts/active_learning_orchestrator.py status
    python scripts/active_learning_orchestrator.py init --bootstrap-relax runs/relax_basic
"""
from __future__ import annotations

import argparse
import hashlib
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

# ── Configuración del orquestador ─────────────────────────────────────────────
MAX_BATCHES = 10
TEST_FRAC = 15            # % held-out (por hash de candidate_id)
CONV_DELTA = 0.0005       # eV/átomo: mejora de test_mae bajo esto 2 batches → convergido
SAT_MIN_NEW = 80          # nuevos-por-batch bajo esto → saturado
ACC_CSV = ROOT / "data" / "processed" / "dft_accumulated.csv"
CURVE_CSV = ROOT / "models" / "surrogate_energy_learning_curve.csv"
STATE_JSON = ROOT / "data" / "batches" / "orchestrator_state.json"
MODEL_PKL = ROOT / "models" / "surrogate_energy.pkl"


def _runs_batches_dir() -> Path:
    from buho.active_learning.batch_loop import BatchLoop

    return BatchLoop(project_root=ROOT).runs_dir


def _is_test(cid: str) -> bool:
    h = int(hashlib.md5(cid.encode()).hexdigest(), 16)
    return (h % 100) < TEST_FRAC


def _load_candidate_index() -> dict:
    idx = {}
    for jl in [ROOT / "data" / "raw" / "generated_candidates.jsonl",
               *sorted((ROOT / "data" / "batches").glob("batch_*/candidates.jsonl"))]:
        if jl.exists():
            for c in HeuristicGenerator.load_jsonl(jl):
                idx[c.candidate_id] = c
    return idx


class Orchestrator:
    def __init__(self):
        self.cand_idx = _load_candidate_index()

    # ── Estado ────────────────────────────────────────────────────────────────
    def load_state(self) -> dict:
        if STATE_JSON.exists():
            return json.loads(STATE_JSON.read_text())
        return {"current_batch": 0, "stopped": False, "reason": None,
                "batches_finalized": [], "max_batches": MAX_BATCHES}

    def save_state(self, s: dict):
        STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
        STATE_JSON.write_text(json.dumps(s, indent=2))

    # ── Acumular resultados DFT (features + energía) ──────────────────────────
    def accumulate(self, relax_dir: Path, batch_tag: str) -> int:
        from buho.dft_jobs.collect_results import ResultCollector
        existing = set()
        if ACC_CSV.exists():
            existing = set(pd.read_csv(ACC_CSV)["candidate_id"].astype(str))
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
            cid = d.name
            if cid in existing:
                continue
            epa = s.get("energy_per_atom_eV")
            c = self.cand_idx.get(cid)
            if epa is None or c is None:
                continue
            feat = ScreeningCascade._features(c)
            feat["candidate_id"] = cid
            feat["formula"] = c.formula
            feat["generation_mode"] = c.generation_mode
            feat["energy_per_atom_eV"] = float(epa)
            feat["batch"] = batch_tag
            feat["split"] = "test" if _is_test(cid) else "train"
            rows.append(feat)
        if not rows:
            return 0
        new = pd.DataFrame(rows)
        if ACC_CSV.exists():
            new = pd.concat([pd.read_csv(ACC_CSV), new], ignore_index=True)
        ACC_CSV.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(ACC_CSV, index=False)
        return len(rows)

    # ── Entrenar + evaluar (held-out) ─────────────────────────────────────────
    def train_eval(self, batch_id: int, n_new: int) -> dict:
        df = pd.read_csv(ACC_CSV)
        # outlier guard robusto sobre energía
        y = df["energy_per_atom_eV"].values.astype(float)
        med = np.median(y); mad = np.median(np.abs(y - med))
        if mad > 0:
            df = df[np.abs(y - med) / (1.4826 * mad) <= 5.0].reset_index(drop=True)
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        feat_cols = list(BASE_FEATURES)
        Xtr, ytr = build_X(tr, feat_cols), tr["energy_per_atom_eV"].values.astype(float)
        Xte, yte = build_X(te, feat_cols), te["energy_per_atom_eV"].values.astype(float)

        m = SurrogateEnsemble().fit(Xtr, ytr, feat_cols)
        ptr = np.array([m.predict_single(x)[0] for x in Xtr])
        pte = np.array([m.predict_single(x)[0] for x in Xte])
        train_mae = float(np.mean(np.abs(ptr - ytr)))
        test_mae = float(np.mean(np.abs(pte - yte)))
        train_rmse = float(np.sqrt(np.mean((ptr - ytr) ** 2)))
        test_rmse = float(np.sqrt(np.mean((pte - yte) ** 2)))
        overfit = round(test_mae / train_mae, 3) if train_mae > 0 else None

        # Guardar predicciones HELD-OUT honestas (modelo train-split sobre test no visto)
        # para el parity/residuos reportables (no contaminado por el modelo de producción).
        heldout = pd.DataFrame({
            "candidate_id": te["candidate_id"].values,
            "formula": te["formula"].values if "formula" in te else te["candidate_id"].values,
            "y_true": yte, "y_pred": pte,
        })
        heldout.to_csv(ROOT / "models" / "surrogate_energy_heldout.csv", index=False)

        # Modelo de producción: entrenar sobre TODO (train+test) y guardar
        Xall = build_X(df, feat_cols)
        yall = df["energy_per_atom_eV"].values.astype(float)
        SurrogateEnsemble().fit(Xall, yall, feat_cols).save(MODEL_PKL)

        rec = {
            "batch": batch_id, "n_total": int(len(df)),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "new_candidates": int(n_new),
            "train_mae": round(train_mae, 5), "test_mae": round(test_mae, 5),
            "train_rmse": round(train_rmse, 5), "test_rmse": round(test_rmse, 5),
            "overfit_ratio": overfit,
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        # append a la curva
        if CURVE_CSV.exists():
            curve = pd.concat([pd.read_csv(CURVE_CSV), pd.DataFrame([rec])], ignore_index=True)
        else:
            curve = pd.DataFrame([rec])
        CURVE_CSV.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(CURVE_CSV, index=False)
        # actualizar gráficas de entrenamiento
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from plot_surrogate_training import make_plots
            make_plots(verbose=False)
        except Exception as e:
            print(f"  (aviso: gráficas no generadas: {e})")
        return rec

    # ── Criterio de paro ──────────────────────────────────────────────────────
    def should_stop(self, batch_id: int, rec: dict) -> tuple[bool, str]:
        if batch_id >= MAX_BATCHES:
            return True, f"max_batches ({MAX_BATCHES}) alcanzado"
        if rec["new_candidates"] < SAT_MIN_NEW:
            return True, f"saturación (nuevos={rec['new_candidates']} < {SAT_MIN_NEW})"
        # convergencia: test_mae mejoró < CONV_DELTA en los últimos 2 batches
        if CURVE_CSV.exists():
            curve = pd.read_csv(CURVE_CSV)
            if len(curve) >= 3:
                last3 = curve["test_mae"].tail(3).values
                if (last3[0] - last3[1]) < CONV_DELTA and (last3[1] - last3[2]) < CONV_DELTA:
                    return True, "convergencia (test_mae estancado 2 batches)"
        return False, ""

    # ── Preparar siguiente batch (NO lanza: el monitor FastAPI lanza el runner) ──
    def launch_batch(self, batch_id: int):
        from buho.active_learning.batch_loop import BatchLoop
        loop = BatchLoop(project_root=ROOT)
        # launch=False → solo genera+criba+prepara jobs DFT; el monitor
        # (poller._check_batch_done) detecta el batch preparado y lanza el runner 5×8.
        return loop.run_batch(batch_id, dry_run=False, launch=False)

    # ── Un paso del loop ──────────────────────────────────────────────────────
    def advance(self) -> dict:
        s = self.load_state()
        if s.get("stopped"):
            print(f"LOOP DETENIDO: {s.get('reason')}")
            return s
        bid = s["current_batch"]
        relax_dir = _runs_batches_dir() / f"batch_{bid:03d}"

        # ¿el DFT del batch actual terminó?
        pend = sum(1 for f in relax_dir.glob("*/status.json")
                   if json.loads(f.read_text()).get("status") == "pending")
        run = sum(1 for f in relax_dir.glob("*/status.json")
                  if json.loads(f.read_text()).get("status") == "running")
        if pend > 0 or run > 0:
            print(f"batch {bid}: DFT aún corriendo (pending={pend}, running={run})")
            return s

        # finalizar batch actual
        self.cand_idx = _load_candidate_index()
        n_new = self.accumulate(relax_dir, f"batch{bid}")
        rec = self.train_eval(bid, n_new)
        s["batches_finalized"].append(bid)
        print(f"[batch {bid}] finalizado: +{n_new} datos | train_mae={rec['train_mae']} "
              f"test_mae={rec['test_mae']} overfit={rec['overfit_ratio']} "
              f"(n_total={rec['n_total']})")

        stop, reason = self.should_stop(bid, rec)
        if stop:
            s["stopped"] = True
            s["reason"] = reason
            self.save_state(s)
            print(f"ÓPTIMO ALCANZADO → PARANDO: {reason}")
            return s

        # lanzar siguiente batch
        nb = bid + 1
        print(f"→ lanzando batch {nb} (generar+cribar+DFT)…")
        m = self.launch_batch(nb)
        s["current_batch"] = nb
        self.save_state(s)
        print(f"[batch {nb}] {m.get('n_selected_dft')} jobs DFT lanzados "
              f"(estado={m.get('status')})")
        return s


def cmd_status():
    o = Orchestrator()
    s = o.load_state()
    print(json.dumps(s, indent=2))
    if CURVE_CSV.exists():
        print("\n=== curva de aprendizaje ===")
        print(pd.read_csv(CURVE_CSV)[
            ["batch", "n_total", "n_train", "n_test", "train_mae", "test_mae",
             "overfit_ratio", "new_candidates"]].to_string(index=False))


def cmd_init(bootstrap_relax: str):
    """Bootstrap: acumula los 500 originales + baseline de la curva (batch=-1).

    Deja `current_batch=0` apuntando al primer batch continuo (runs/batches/batch_000).
    """
    o = Orchestrator()
    n = o.accumulate(Path(bootstrap_relax), "bootstrap")
    print(f"Bootstrap: +{n} datos acumulados de {bootstrap_relax}")
    rec = o.train_eval(-1, n)   # punto baseline de la curva
    print(f"baseline surrogate: train_mae={rec['train_mae']} test_mae={rec['test_mae']} "
          f"overfit={rec['overfit_ratio']} (n_total={rec['n_total']})")
    s = o.load_state()
    s["current_batch"] = 0
    o.save_state(s)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("advance")
    sub.add_parser("status")
    pi = sub.add_parser("init")
    pi.add_argument("--bootstrap-relax", default=str(ROOT / "runs" / "relax_basic"))
    args = ap.parse_args()
    if args.cmd == "advance":
        Orchestrator().advance()
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "init":
        cmd_init(args.bootstrap_relax)


if __name__ == "__main__":
    main()
