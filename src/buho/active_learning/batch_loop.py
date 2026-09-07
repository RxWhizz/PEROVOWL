"""Driver de active learning por batches para descubrimiento ABX3.

Convierte el pipeline de una pasada en un loop por batches auto-mejorante:

  run_batch(N):
    1. generar batch continuo reproducible (seed = random_seed + N)
    2. cascada de cribado barato (Tier 0-2: descriptores + surrogate + MLFF Eform)
    3. seleccionar top-N por score de adquisición (filtrando estabilidad)
    4. preparar jobs DFT single-point para los seleccionados (GPAW 24.6, domain=8)
    [DFT corre vía scripts/buho_relax_runner.py — proceso aparte]

  finalize_batch(N):
    5. recolectar resultados DFT (con guard de outliers / trusted_label)
    6. anexar filas confiables a data/processed/surrogate_training_dft.csv
    7. (opcional) reentrenar el surrogate → guía los siguientes batches

Artefactos por batch en data/batches/batch_NNN/:
  candidates.jsonl, cascade_scores.csv, selected_for_dft.csv,
  dft_results.csv, batch_manifest.json
Jobs DFT en runs/batches/batch_NNN/.

Reusa: HeuristicGenerator.generate_batch, ScreeningCascade,
RelaxationJobPreparer, ResultCollector, ml_surrogate (fit/save).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from buho.generator.heuristic_generator import HeuristicGenerator
from buho.screening.cascade import ScreeningCascade


class BatchLoop:
    def __init__(self, config_path: str | Path = "config/generator.yaml",
                 project_root: Optional[Path] = None):
        self._root = Path(project_root) if project_root else ROOT
        self._config_path = Path(config_path)
        self._cfg = yaml.safe_load(self._config_path.read_text())
        self._scr = self._cfg.get("screening", {})
        self._gen = HeuristicGenerator(str(self._config_path))
        self._cascade = ScreeningCascade(self._cfg, project_root=self._root)
        paths = self._cfg.get("paths", {})
        self._batches_dir = self._resolve_path(paths.get("batch_artifacts_dir", "data/batches"))
        self._runs_dir = self._resolve_path(paths.get("runs_batches_dir", "runs/batches"))

    # ── Utilidades ────────────────────────────────────────────────────────────
    def _resolve_path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self._root / p

    @property
    def runs_dir(self) -> Path:
        return self._runs_dir

    def _batch_dir(self, batch_id: int) -> Path:
        d = self._batches_dir / f"batch_{batch_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _write_manifest(bdir: Path, manifest: dict) -> None:
        (bdir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Paso 1-4: generar → cribar → seleccionar → preparar DFT ───────────────
    def run_batch(self, batch_id: int, dry_run: bool = False,
                  launch: bool = False) -> dict:
        bdir = self._batch_dir(batch_id)
        t0 = datetime.utcnow().isoformat() + "Z"

        # 1. Generar (con dedup persistente entre batches)
        registry = self._scr.get("registry_path")
        registry = str(self._root / registry) if registry else None
        cands = self._gen.generate_batch(batch_id, registry_path=registry)
        HeuristicGenerator.save_jsonl(cands, bdir / "candidates.jsonl")

        # 2. Cascada de cribado (Tier 0-2; MLFF es cribado barato, corre siempre)
        df = self._cascade.screen(cands, run_mlff=self._scr.get("tier2_mlff", True))
        df.to_csv(bdir / "cascade_scores.csv", index=False)

        # 3. Seleccionar para DFT (n_dft<=0 → todos los que pasen estabilidad)
        n_dft = int(self._scr.get("n_dft_per_batch", 0))
        sel = self._cascade.select_for_dft(df, n=n_dft)
        sel.to_csv(bdir / "selected_for_dft.csv", index=False)

        manifest = {
            "batch_id": batch_id,
            "started_at": t0,
            "n_generated": len(cands),
            "n_passed_filters": int(len(df)),
            "n_passed_stability": int(df["passed_eform"].sum()) if len(df) else 0,
            "n_selected_dft": int(len(sel)),
            "fraction_mode": self._gen._fraction_mode,
            "status": "screened",
        }

        if dry_run:
            manifest["status"] = "screened_dry_run"
            self._write_manifest(bdir, manifest)
            print(f"[batch {batch_id}] DRY-RUN: {len(cands)} generados → "
                  f"{len(df)} pasan filtros → {len(sel)} seleccionados "
                  f"(sin DFT). Artefactos en {bdir}")
            return manifest

        # 4. Preparar jobs DFT para los seleccionados
        from buho.dft_jobs.prepare_relaxation_jobs import RelaxationJobPreparer
        sel_ids = set(sel["candidate_id"])
        sel_cands = [c for c in cands if c.candidate_id in sel_ids]
        relax_dir = self._runs_dir / f"batch_{batch_id:03d}"
        preparer = RelaxationJobPreparer(self._cfg, project_root=self._root, n_cores=8)
        prepared = preparer.prepare(sel_cands, out_root=relax_dir,
                                    config_src=self._config_path)
        manifest["status"] = "dft_prepared"
        manifest["n_dft_prepared"] = len(prepared)
        manifest["relax_dir"] = str(relax_dir)
        self._write_manifest(bdir, manifest)
        print(f"[batch {batch_id}] {len(sel_cands)} jobs DFT preparados en {relax_dir}")

        if launch:
            self._launch_runner(relax_dir)
            manifest["status"] = "dft_running"
            self._write_manifest(bdir, manifest)
        return manifest

    def _launch_runner(self, relax_dir: Path) -> None:
        cmd = [sys.executable, str(self._root / "scripts" / "buho_relax_runner.py"),
               "--slots", "5", "--cores", "8", "--stagger", "8",
               "--relax-dir", str(relax_dir)]
        subprocess.Popen(cmd, cwd=str(self._root),
                         stdout=open(relax_dir / "runner.out", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        print(f"  runner lanzado (5×8) sobre {relax_dir}")

    # ── Paso 5-7: recolectar → anexar → reentrenar ────────────────────────────
    def finalize_batch(self, batch_id: int, retrain: bool = False) -> dict:
        bdir = self._batch_dir(batch_id)
        manifest = json.loads((bdir / "batch_manifest.json").read_text())
        relax_dir = Path(manifest.get("relax_dir",
                                      self._runs_dir / f"batch_{batch_id:03d}"))

        from buho.dft_jobs.collect_results import ResultCollector
        collector = ResultCollector(project_root=self._root)
        res = collector.collect_all(relax_dir)
        res.to_csv(bdir / "dft_results.csv", index=False)

        # Unir resultados DFT con scores de cascada (features + Eform MLFF)
        casc = pd.read_csv(bdir / "cascade_scores.csv")
        merged = res.merge(casc, on="candidate_id", how="left", suffixes=("", "_casc"))
        trusted = merged[merged.get("trusted_label", True) == True]  # noqa: E712

        manifest["n_dft_collected"] = int(len(res))
        manifest["n_dft_converged"] = int(res["converged"].sum()) if "converged" in res else 0
        manifest["n_trusted"] = int(len(trusted))
        manifest["finalized_at"] = datetime.utcnow().isoformat() + "Z"

        n_appended = self._append_training(trusted, batch_id)
        manifest["n_appended_training"] = n_appended
        manifest["status"] = "collected"

        if retrain and n_appended:
            metrics = self._retrain()
            manifest["retrain_metrics"] = metrics
            manifest["status"] = "complete"

        self._write_manifest(bdir, manifest)
        print(f"[batch {batch_id}] recolectado: {manifest['n_dft_converged']} "
              f"convergidos, {n_appended} anexados a training"
              f"{', reentrenado' if retrain and n_appended else ''}")
        return manifest

    def _append_training(self, trusted: pd.DataFrame, batch_id: int) -> int:
        """Anexa filas DFT confiables al CSV de training DFT-acumulado.

        Target = bandgap_preliminary_eV (gap DFT single-point, ruidoso pero
        consistente). Features = descriptores de cascada + Eform MLFF.
        """
        if trusted.empty:
            return 0
        acc_path = self._root / "data" / "processed" / "surrogate_training_dft.csv"
        rows = []
        for _, r in trusted.iterrows():
            eg = r.get("bandgap_preliminary_eV")
            if eg is None or pd.isna(eg):
                continue  # sin target utilizable
            rows.append({
                "material_id": r["candidate_id"],
                "tolerance_t": r.get("tolerance_t"),
                "oct_factor": r.get("oct_factor"),
                "vol_est_A3": r.get("vol_est_A3"),
                "Eform_eV_atom": r.get("Eform_eV_atom"),
                "band_gap_gga_eV": eg,
                "energy_per_atom_eV": r.get("energy_per_atom_eV"),
                "Eg_target_eV": eg,
                "source": f"buho_dft_batch{batch_id}",
            })
        if not rows:
            return 0
        new = pd.DataFrame(rows)
        if acc_path.exists():
            new = pd.concat([pd.read_csv(acc_path), new], ignore_index=True)
        acc_path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(acc_path, index=False)
        return len(rows)

    def _retrain(self) -> dict:
        """Reentrena el surrogate de bandgap con literatura + DFT acumulado.

        Construye features completos por composición para las filas DFT (que solo
        guardan descriptores parciales) reconstruyéndolos desde el candidate;
        usa BASE_FEATURES disponibles. Respalda el modelo anterior.
        """
        import numpy as np
        from ml_surrogate.features import BASE_FEATURES, build_X
        from ml_surrogate.model import SurrogateEnsemble

        lit_path = self._root / "data" / "surrogate_training.csv"
        dft_path = self._root / "data" / "processed" / "surrogate_training_dft.csv"
        frames = []
        if lit_path.exists():
            frames.append(pd.read_csv(lit_path))
        if dft_path.exists():
            frames.append(pd.read_csv(dft_path))
        if not frames:
            return {"error": "no training data"}
        df = pd.concat(frames, ignore_index=True)

        # Columnas de features presentes (intersección con BASE+opcionales)
        feat_cols = [c for c in BASE_FEATURES if c in df.columns]
        # `band_gap_gga_eV` se excluye por la misma razon que en
        # `discovery.engine._retrain_bandgap`: aqui vale lo mismo que
        # `Eg_target_eV`, asi que seria entrenar a copiar el target, y en
        # inferencia no existe.
        for opt in ("a_lat_mp_A", "Eform_eV_atom"):
            if opt in df.columns:
                feat_cols.append(opt)
        df = df.dropna(subset=["Eg_target_eV"])
        X = build_X(df, feat_cols)
        y = df["Eg_target_eV"].values.astype(float)

        model_path = self._root / "models" / "surrogate_bandgap.pkl"
        if model_path.exists():
            backup = model_path.with_suffix(".pkl.bak")
            backup.write_bytes(model_path.read_bytes())
        model = SurrogateEnsemble().fit(X, y, feat_cols)
        model.save(model_path)
        # invalidar cache del surrogate en la cascada
        self._cascade._surrogate = None
        return {"n_train": int(len(y)), "n_features": len(feat_cols)}
