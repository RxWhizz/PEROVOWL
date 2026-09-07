"""Persistent autonomous discovery loop for PEROVOWL."""

from __future__ import annotations

import hashlib
import copy
import json
import logging
import math
import os
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from buho import bandgap_scissor, eg_scale
from buho.discovery.pareto import pareto_front
from buho.discovery.space import ChemicalSpaceEnumerator, fraction_grid
from buho.generator.heuristic_generator import GeneratedCandidate, HeuristicGenerator
from buho.mlff_runtime import MLFFUnavailableError
from buho.screening.cascade import ScreeningCascade
from buho.structure.occupancy import clave_estructura

ROOT = Path(__file__).resolve().parents[3]

log = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = {"pending", "running", "stalled", "oscillating"}
DFT_LEDGER_STATUSES = {
    "dft_selected",
    "dft_prepared",
    "dft_running",
    "dft_converged",
    "dft_failed",
    "training_added",
}
TEXT_LEDGER_COLUMNS = {
    "candidate_id",
    "riesgo_politipo",
    "scores_imputados",
    "formula",
    "generation_mode",
    "A_site",
    "B_site",
    "X_site",
    "B_family",
    "dominant_B",
    "dominant_halide",
    "status",
    "round_selected",
    "round_completed",
    "last_screened_round",
    "dropped_at_tier",
    "drop_reason",
    "error_message",
    "job_id",
    "dft_status",
}
BOOLEAN_LEDGER_COLUMNS = {
    "converged",
    "in_pv_window",
    "is_organic_A",
    "is_stable",
    "mlff_evaluated",
    "passed_eform",
    "trusted_label",
}
RUNNER_ENV_KEYS = {
    "backend": ("BUHO_DFT_BACKEND",),
    "bash": ("BUHO_BASH",),
    "conda_bin": ("BUHO_CONDA_BIN", "CONDA_EXE"),
    "conda_env": ("BUHO_GPAW_CONDA_ENV", "GPAW_CONDA_ENV"),
    "launcher": ("BUHO_DFT_LAUNCHER",),
    "mpirun": ("BUHO_MPI_LAUNCHER", "MPIEXEC", "MPIRUN"),
    "python": ("BUHO_GPAW_PYTHON", "GPAW_PYTHON"),
    "setup_path": ("BUHO_GPAW_SETUP_PATH", "GPAW_SETUP_PATH"),
}
WSL_RUNNER_ENV_KEYS = {
    "bash": ("BUHO_WSL_BASH", "BUHO_BASH"),
    "conda_bin": ("BUHO_WSL_CONDA_BIN", "BUHO_CONDA_BIN"),
    "conda_env": ("BUHO_WSL_GPAW_CONDA_ENV", "BUHO_GPAW_CONDA_ENV", "GPAW_CONDA_ENV"),
    "distro": ("BUHO_WSL_DISTRO",),
    "driver_python": ("BUHO_WSL_DRIVER_PYTHON",),
    "launcher": ("BUHO_WSL_DFT_LAUNCHER", "BUHO_DFT_LAUNCHER"),
    "mpirun": ("BUHO_WSL_MPI_LAUNCHER", "BUHO_MPI_LAUNCHER", "MPIEXEC", "MPIRUN"),
    "project_root": ("BUHO_WSL_PROJECT_ROOT",),
    "python": ("BUHO_WSL_GPAW_PYTHON", "BUHO_GPAW_PYTHON", "GPAW_PYTHON"),
    "script_path": ("BUHO_WSL_RUNNER_SCRIPT",),
    "setup_path": ("BUHO_WSL_GPAW_SETUP_PATH", "BUHO_GPAW_SETUP_PATH", "GPAW_SETUP_PATH"),
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


#: Particiones de la validación cruzada del surrogate. Cinco es el compromiso
#: habitual, y con las decenas de muestras que produce una ronda cada pliegue
#: sigue teniendo suficientes puntos para que el MAE no sea ruido.
CV_FOLDS = 5


def _cv_metrics(X: Any, y: Any, feat_cols: list[str],  # noqa: N803 - X matriz / y vector, como el resto del fichero
                *, folds: int = CV_FOLDS) -> dict[str, Any]:
    """MAE fuera de muestra del surrogate, y el de predecir la media.

    `train_mae_eV` se calcula sobre las mismas filas del ajuste, así que **baja**
    al crecer el conjunto: con cinco features y un ensemble de árboles, ochenta
    puntos se memorizan casi exactos. Leerlo como "el modelo mejora" es leerlo al
    revés. Esto entrena en k-1 pliegues y mide en el que queda fuera, que es lo
    único que dice si generaliza.

    `baseline_mae_eV` es el MAE de predecir siempre la media del entrenamiento.
    Es la vara: si `cv_mae_eV` no baja de ahí, el surrogate no está aportando
    nada sobre no tener modelo, por bonito que sea su error de ajuste.
    """
    from ml_surrogate.model import SurrogateEnsemble

    n = int(len(y))
    if n < 2 * folds:
        # Con pliegues de un par de puntos el MAE es ruido; mejor decir que no
        # se pudo medir que publicar un número que no significa nada.
        return {
            "cv_mae_eV": None,
            "cv_folds": None,
            "baseline_mae_eV": None,
            "cv_skipped_reason": f"hacen falta {2 * folds} muestras y hay {n}",
        }

    from sklearn.model_selection import KFold

    errores: list[float] = []
    base: list[float] = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        try:
            m = SurrogateEnsemble().fit(X[train_idx], y[train_idx], feat_cols)
            pred, _ = m.predict_batch(X[test_idx])
        except Exception:
            # Un pliegue que no converge no debe tumbar la ronda entera: el
            # reentrenamiento ya ocurrió y el modelo está guardado.
            continue
        errores.extend(np.abs(np.asarray(pred) - y[test_idx]).tolist())
        base.extend(np.abs(y[train_idx].mean() - y[test_idx]).tolist())

    if not errores:
        return {
            "cv_mae_eV": None,
            "cv_folds": None,
            "baseline_mae_eV": None,
            "cv_skipped_reason": "ningún pliegue pudo evaluarse",
        }

    return {
        "cv_mae_eV": round(float(np.mean(errores)), 5),
        "cv_folds": int(folds),
        "baseline_mae_eV": round(float(np.mean(base)), 5),
    }


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _bool_or_na(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return pd.NA
    if text in {"true", "1", "yes", "y", "si", "sí"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return value


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _split_for_candidate(candidate_id: str) -> str:
    digest = int(hashlib.sha1(candidate_id.encode()).hexdigest()[:8], 16)
    return "test" if digest % 100 < 15 else "train"


def _dominant(fracs: dict[str, float]) -> str:
    return max(fracs, key=lambda key: float(fracs[key])) if fracs else ""


def _family(fracs: dict[str, float]) -> str:
    present = [sp for sp, frac in sorted(fracs.items()) if float(frac) > 0.01]
    return "".join(present) if present else "other"


def _transport_score(m_e: Any, m_h: Any) -> float:
    vals = [_finite(m_e), _finite(m_h)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.5
    scores = [1.0 / (1.0 + math.exp(5.0 * (v - 0.5))) for v in vals]
    return float(sum(scores) / len(scores))


def _dielectric_score(eps: Any) -> float:
    eps_f = _finite(eps)
    if eps_f is None or eps_f <= 0:
        return 0.5
    return float(min(1.0, eps_f / (eps_f + 10.0)))


def _exciton_score(m_e: Any, m_h: Any, eps: Any) -> tuple[float, float | None]:
    e, h, er = _finite(m_e), _finite(m_h), _finite(eps)
    if e is None or h is None or er is None or er <= 0:
        return 0.5, None
    reduced = (e * h) / (e + h)
    eb_eV = 13.60570 * reduced / (er ** 2)
    score = 1.0 / (1.0 + math.exp(50.0 * (eb_eV - 0.075)))
    return float(score), float(eb_eV * 1000.0)


class DiscoveryLoop:
    """Run an autonomous ML discovery -> DFT -> retrain loop."""

    def __init__(
        self,
        config_path: str | Path | dict[str, Any] = "config/generator.yaml",
        *,
        project_root: Path | None = None,
        data_root: Path | None = None,
        models_root: Path | None = None,
        bundle_root: Path | None = None,
        config_source_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root else ROOT
        self.data_root = Path(data_root) if data_root else self.project_root
        if isinstance(config_path, dict):
            self.config = copy.deepcopy(config_path)
            source = Path(config_source_path or "config/generator.yaml")
            self.config_path = source if source.is_absolute() else self.project_root / source
        else:
            cfg_path = Path(config_path)
            self.config_path = cfg_path if cfg_path.is_absolute() else self.project_root / cfg_path
            with self.config_path.open(encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh) or {}

        self.discovery = self.config.get("discovery", {}) or {}
        # `models_root` es donde se ESCRIBE (raiz de datos del usuario).
        # `bundle_root` es solo de lectura: los modelos de fabrica viajan
        # dentro del binario y ahi no se puede ni se debe escribir.
        self.models_root = Path(models_root) if models_root else self.project_root
        self.bundle_root = Path(bundle_root) if bundle_root else None

        paths = self.config.get("paths", {}) or {}
        self.output_dir = self._resolve(
            self.discovery.get("output_dir")
            or paths.get("discovery_dir")
            or "data/discovery"
        )
        self.rounds_dir = self.output_dir / "rounds"
        self.dft_runs_dir = self._resolve(paths.get("runs_batches_dir", "runs/batches")) / "discovery"
        self.candidates_path = self.output_dir / "candidates.jsonl"
        self.ledger_path = self.output_dir / "ledger.csv"
        self.state_path = self.output_dir / "state.json"
        self.frontier_path = self.output_dir / "frontier.csv"
        self.training_path = self.output_dir / "surrogate_training_dft.csv"
        self.metrics_path = self.output_dir / "model_metrics.jsonl"

        self.batch_size = int(self.discovery.get("dft_per_round", 30))
        self.mlff_pool_size = int(self.discovery.get("mlff_pool_size", 5000))
        self.frontier_size = int(self.discovery.get("frontier_size", 500))
        self.pareto_input_size = int(self.discovery.get("pareto_input_size", 5000))
        self.min_pv_score = float(self.discovery.get("min_pv_score", 0.45))
        self.require_mlff_for_dft = bool(self.discovery.get("require_mlff_for_dft", True))
        self.poll_interval_sec = int(self.discovery.get("poll_interval_sec", 60))
        self.runner_slots = int(self.discovery.get("runner_slots", 5))
        self.runner_cores = int(self.discovery.get("runner_cores", 8))
        self.runner_stagger = int(self.discovery.get("runner_stagger", 8))
        self.allow_windows_runner = bool(self.discovery.get("allow_windows_runner", False))
        self.runner_backend = str(self.discovery.get("runner_backend", "auto")).lower()
        self.runner_launcher = str(self.discovery.get("runner_launcher", "auto")).lower()
        self.runner_preflight_timeout = int(self.discovery.get("runner_preflight_timeout", 90))
        self.runner_wsl = self.discovery.get("wsl", {}) or {}

    # ── Paths and state ────────────────────────────────────────────────────

    def _resolve(self, value: str | Path) -> Path:
        p = Path(self._translate_posix_mount(str(value))).expanduser()
        return p if p.is_absolute() else self.data_root / p

    def _translate_posix_mount(self, raw: str) -> str:
        if sys.platform != "win32" or not raw.startswith("/"):
            return raw

        mappings = list(self.discovery.get("windows_mounts") or [])
        env_root = os.environ.get("PEROVOWL_POSIX_MOUNT_ROOT") or os.environ.get("DFT_POSIX_MOUNT_ROOT")
        if env_root:
            mappings.insert(
                0,
                {
                    # Sin default: era la ruta del disco externo de una
                    # maquina concreta, inutil en cualquier otra.
                    "posix": self.discovery.get("posix_mount_prefix", ""),
                    "windows": env_root,
                },
            )

        for mapping in mappings:
            posix_root = str(mapping.get("posix", "")).rstrip("/")
            windows_root = str(mapping.get("windows", "")).rstrip("/\\")
            if not posix_root or not windows_root:
                continue
            if raw == posix_root or raw.startswith(posix_root + "/"):
                suffix = raw[len(posix_root):].lstrip("/")
                parts = [part for part in suffix.split("/") if part]
                return str(Path(windows_root).joinpath(*parts))

        return raw

    def _runner_backend_name(self) -> str:
        backend = (_first_env(RUNNER_ENV_KEYS["backend"]) or self.runner_backend).lower()
        if backend not in {"auto", "local", "wsl"}:
            raise RuntimeError(f"Backend DFT no reconocido: {backend}")
        if backend != "auto":
            return backend
        if sys.platform == "win32" and not self.allow_windows_runner:
            return "wsl"
        return "local"

    def _wsl_mount_mappings(self) -> list[dict[str, str]]:
        mappings: list[dict[str, str]] = []
        for raw in self.runner_wsl.get("mounts") or self.discovery.get("wsl_mounts") or []:
            if raw.get("windows") and raw.get("wsl"):
                mappings.append({"windows": str(raw["windows"]), "wsl": str(raw["wsl"])})
        for raw in self.discovery.get("windows_mounts") or []:
            if raw.get("windows") and raw.get("wsl"):
                mappings.append({"windows": str(raw["windows"]), "wsl": str(raw["wsl"])})
        mappings.append({"windows": str(self.project_root), "wsl": self._default_wsl_drive_path(self.project_root)})
        return mappings

    @staticmethod
    def _default_wsl_drive_path(path: str | Path) -> str:
        raw = str(path).replace("\\", "/")
        if len(raw) >= 2 and raw[1] == ":":
            drive = raw[0].lower()
            rest = raw[2:].lstrip("/")
            return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
        return raw

    def _windows_path_to_wsl(self, path: str | Path) -> str:
        raw = str(path).replace("\\", "/")
        raw_cmp = raw.rstrip("/").lower()
        for mapping in self._wsl_mount_mappings():
            windows_root = str(mapping["windows"]).replace("\\", "/").rstrip("/")
            windows_cmp = windows_root.lower()
            if raw_cmp == windows_cmp or raw_cmp.startswith(windows_cmp + "/"):
                suffix = raw[len(windows_root):].lstrip("/")
                wsl_root = str(mapping["wsl"]).rstrip("/")
                return f"{wsl_root}/{suffix}" if suffix else wsl_root
        return self._default_wsl_drive_path(raw)

    def _runner_option(self, key: str, backend: str) -> str | None:
        value = None
        env_keys = WSL_RUNNER_ENV_KEYS.get(key) if backend == "wsl" else RUNNER_ENV_KEYS.get(key)
        if env_keys:
            value = _first_env(env_keys)
        if backend == "wsl":
            value = value or self.runner_wsl.get(key) or self.runner_wsl.get(f"runner_{key}")
        if value is None:
            value = self.discovery.get(f"runner_{key}") or self.discovery.get(key)
        if value is None:
            return None
        text = str(value)
        if backend == "wsl" and len(text) >= 2 and text[1] == ":":
            return self._windows_path_to_wsl(text)
        return text

    def _runner_script_args(self, runs_dir: Path, backend: str, *, preflight_only: bool = False) -> list[str]:
        runner_runs_dir = self._windows_path_to_wsl(runs_dir) if backend == "wsl" else str(runs_dir)
        launcher = self._runner_option("launcher", backend) or self.runner_launcher
        args = [
            "--slots",
            str(self.runner_slots),
            "--cores",
            str(self.runner_cores),
            "--stagger",
            str(self.runner_stagger),
            "--relax-dir",
            runner_runs_dir,
            "--launcher",
            launcher,
        ]
        options = [
            ("conda_bin", "--conda-bin"),
            ("conda_env", "--conda-env"),
            ("python", "--python"),
            ("mpirun", "--mpirun"),
            ("setup_path", "--setup-path"),
            ("bash", "--bash"),
        ]
        for key, flag in options:
            value = self._runner_option(key, backend)
            if value:
                args.extend([flag, value])
        if preflight_only:
            args.append("--preflight-only")
        return args

    def _runner_command(self, runs_dir: Path, *, preflight_only: bool = False) -> tuple[str, list[str], str]:
        backend = self._runner_backend_name()
        if backend == "wsl":
            project_root = str(self._runner_option("project_root", backend) or self._windows_path_to_wsl(self.project_root))
            script = str(self._runner_option("script_path", backend) or f"{project_root}/scripts/buho_relax_runner.py")
            driver_python = str(self._runner_option("driver_python", backend) or "python3")
            argv = [driver_python, script, *self._runner_script_args(runs_dir, backend, preflight_only=preflight_only)]
            shell_line = f"cd {shlex.quote(project_root)} && " + " ".join(shlex.quote(arg) for arg in argv)
            cmd = ["wsl.exe"]
            distro = self._runner_option("distro", backend) or self.discovery.get("wsl_distro")
            if distro:
                cmd.extend(["-d", str(distro)])
            cmd.extend(["--", "bash", "-lc", shell_line])
            return backend, cmd, str(self.project_root)

        script = self.project_root / "scripts" / "buho_relax_runner.py"
        driver_python = str(self.discovery.get("runner_driver_python") or sys.executable)
        cmd = [driver_python, str(script), *self._runner_script_args(runs_dir, backend, preflight_only=preflight_only)]
        return backend, cmd, str(self.project_root)

    @staticmethod
    def _detached_kwargs() -> dict[str, Any]:
        if sys.platform == "win32":
            flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            return {"creationflags": flag} if flag else {}
        return {"start_new_session": True}

    def _round_dir(self, round_id: int) -> Path:
        return self.rounds_dir / f"round_{round_id:03d}"

    def _raices_modelos(self) -> list[Path]:
        """Dónde buscar modelos, en orden: primero los datos, luego el bundle.

        Se escribe siempre en `models_root`. Los de fábrica viajan dentro del
        binario (`bundle_root`), que en PyInstaller es el directorio de
        extracción: leerlo está bien, escribirlo no — se pierde al actualizar y
        falla si la app quedó instalada en un sitio de solo lectura.
        """
        raices = [self.models_root]
        if self.bundle_root and self.bundle_root.resolve() != self.models_root.resolve():
            raices.append(self.bundle_root)
        return raices

    def _buscar_modelo(self, *partes: str) -> Path | None:
        """Primer modelo que exista recorriendo las raíces. `None` si ninguno."""
        for raiz in self._raices_modelos():
            candidato = raiz.joinpath(*partes)
            if candidato.is_file():
                return candidato
        return None

    def _round_runs_dir(self, round_id: int) -> Path:
        return self.dft_runs_dir / f"round_{round_id:03d}"

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {
                "status": "not_initialized",
                "current_round": 0,
                "created_at": None,
                "updated_at": None,
                "stop_reason": None,
            }
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "status": "error",
                "current_round": 0,
                "error": f"state corrupto: {self.state_path}",
                "updated_at": _utc(),
            }

    def _save_state(self, state: dict[str, Any]) -> None:
        state = dict(state)
        state["updated_at"] = _utc()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _read_ledger(self) -> pd.DataFrame:
        if not self.ledger_path.is_file():
            return pd.DataFrame()
        df = pd.read_csv(self.ledger_path, low_memory=False)
        df = self._coerce_ledger_columns(df)
        return df

    @staticmethod
    def _coerce_ledger_columns(df: pd.DataFrame) -> pd.DataFrame:
        if "status" not in df:
            df["status"] = "unseen"
        for col in TEXT_LEDGER_COLUMNS:
            if col in df:
                df[col] = df[col].fillna("").astype(object)
        for col in BOOLEAN_LEDGER_COLUMNS:
            if col in df:
                df[col] = df[col].map(_bool_or_na).astype(object)
        df["status"] = df["status"].replace("", "unseen").fillna("unseen").astype(object)
        return df

    def _write_ledger(self, df: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.ledger_path, index=False)

    def _load_candidates(self) -> dict[str, GeneratedCandidate]:
        if not self.candidates_path.is_file():
            return {}
        candidates = HeuristicGenerator.load_jsonl(self.candidates_path)
        return {c.candidate_id: c for c in candidates}

    # ── Initialization ─────────────────────────────────────────────────────

    def init_space(self, *, reset: bool = False) -> dict[str, Any]:
        """Enumerate the finite space and create an initial ledger."""
        if self.ledger_path.exists() and not reset:
            return self.status()

        enumerator = ChemicalSpaceEnumerator(self.config_path)
        candidates, stats = enumerator.enumerate(physical_viable_only=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        HeuristicGenerator.save_jsonl(candidates, self.candidates_path)

        rows = [self._candidate_record(c) for c in candidates]
        ledger = pd.DataFrame(rows)
        self._write_ledger(ledger)

        state = {
            "status": "idle",
            "current_round": 0,
            "created_at": _utc(),
            "stop_reason": None,
            "space": stats.as_dict(),
            "dft_per_round": self.batch_size,
            "paths": self._paths_payload(),
        }
        self._save_state(state)
        self._write_frontier(pd.DataFrame())
        return self.status()

    def _candidate_record(self, c: GeneratedCandidate) -> dict[str, Any]:
        b_fracs = c.fractions.get("B", {})
        x_fracs = c.fractions.get("X", {})
        return {
            "candidate_id": c.candidate_id,
            "formula": c.formula,
            "generation_mode": c.generation_mode,
            "A_site": "+".join(c.A_site_species),
            "B_site": "+".join(c.B_site_species),
            "X_site": "+".join(c.X_site_species),
            "B_family": _family(b_fracs),
            "dominant_B": _dominant(b_fracs),
            "dominant_halide": _dominant(x_fracs),
            "tolerance_t": c.tolerance_t,
            "oct_factor": c.oct_factor,
            "vol_est_A3": c.vol_est_A3,
            "is_organic_A": c.is_organic_A,
            "status": "unseen",
            "round_selected": "",
            "round_completed": "",
            "last_screened_round": "",
            "drop_reason": "",
        }

    def _paths_payload(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "ledger": str(self.ledger_path),
            "frontier": str(self.frontier_path),
            "dft_runs_dir": str(self.dft_runs_dir),
            "training": str(self.training_path),
        }

    def _config_payload(self) -> dict[str, Any]:
        cs = self.config.get("chemical_space", {}) or {}
        gen = self.config.get("generation", {}) or {}
        modes = gen.get("modes", {}) or {}
        space = self.discovery.get("space", {}) or {}
        min_fraction = float(space.get("min_fraction", 0.05))
        max_fraction = float(space.get("max_fraction", 0.95))
        fraction_step = float(space.get("fraction_step", 0.01))
        return {
            "A_sites": list(cs.get("A_sites", [])),
            "B_sites": list(cs.get("B_sites", [])),
            "X_sites": list(cs.get("X_sites", [])),
            "modes": {
                "pure": bool(modes.get("pure", True)),
                "A_mixed": bool(modes.get("A_mixed", True)),
                "B_mixed": bool(modes.get("B_mixed", True)),
                "X_mixed": bool(modes.get("X_mixed", True)),
                "multi_mixed": bool(modes.get("multi_mixed", False)),
            },
            "min_fraction": min_fraction,
            "max_fraction": max_fraction,
            "fraction_step": fraction_step,
            "fraction_values": fraction_grid(min_fraction, max_fraction, fraction_step),
            "include_multi_mixed": bool(space.get("include_multi_mixed", False)),
            "dft_per_round": self.batch_size,
            "runner_backend": self.runner_backend,
            "runner_effective_backend": self._runner_backend_name(),
            "runner_launcher": self.runner_launcher,
        }

    # ── Scoring ────────────────────────────────────────────────────────────

    def score_space(self, *, use_mlff: bool | None = None) -> dict[str, Any]:
        """Score all non-final candidates and refresh the Pareto frontier.

        `_score_space_impl` marca el estado como "screening" al empezar y solo
        vuelve a "idle" al terminar. Si algo revienta en medio, el estado
        persistido se queda en "screening" y el protocolo no vuelve a arrancar
        nunca: parece que sigue cribando cuando en realidad el hilo murió. Por
        eso cualquier fallo se anota como "error" antes de propagarse.
        """
        try:
            return self._score_space_impl(use_mlff=use_mlff)
        except Exception as exc:
            state = self._load_state()
            if state.get("status") == "screening":
                state["status"] = "error"
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._save_state(state)
            raise

    def _score_space_impl(self, *, use_mlff: bool | None = None) -> dict[str, Any]:
        if not self.ledger_path.exists():
            self.init_space()

        state = self._load_state()
        round_id = int(state.get("current_round", 0))
        state["status"] = "screening"
        self._save_state(state)

        candidates = self._load_candidates()
        ledger = self._read_ledger()
        excluded = set(ledger[ledger["status"].isin(DFT_LEDGER_STATUSES)]["candidate_id"])
        work_candidates = [c for cid, c in candidates.items() if cid not in excluded]

        cascade = ScreeningCascade(self.config, project_root=self.models_root,
                                   extra_roots=self._raices_modelos()[1:])
        df_light = cascade.screen(work_candidates, run_mlff=False)
        df_light["mlff_evaluated"] = False

        requested_mlff = (
            bool((self.config.get("screening", {}) or {}).get("tier2_mlff", True))
            if use_mlff is None else bool(use_mlff)
        )
        df = df_light
        n_mlff = 0
        mlff_warning: dict[str, str] | None = None
        if requested_mlff and not df_light.empty and self.mlff_pool_size > 0:
            pool_ids = (
                df_light[df_light["dropped_at_tier"].isna()]
                .sort_values("total_score", ascending=False)
                .head(self.mlff_pool_size)["candidate_id"]
                .astype(str)
                .tolist()
            )
            pool_candidates = [candidates[cid] for cid in pool_ids if cid in candidates]
            if pool_candidates:
                try:
                    df_mlff = cascade.screen(pool_candidates, run_mlff=True)
                except MLFFUnavailableError as exc:
                    # Que falte el entorno MLFF no puede matar la ronda: Tier 0/1
                    # ya deja un ranking utilizable y el DFT es lo caro. Se anota
                    # el motivo, se sigue sin Tier 2 y la GUI lo enseña; antes
                    # esto reventaba el hilo y dejaba el estado en "screening"
                    # para siempre.
                    mlff_warning = {"error": str(exc), "remediation": exc.remediation}
                    requested_mlff = False
                else:
                    df_mlff["mlff_evaluated"] = True
                    df = self._replace_rows(df_light, df_mlff)
                    n_mlff = len(df_mlff)

        df = self._attach_property_predictions(df)
        df = self._score_objectives(df)
        df = self._attach_ledger_metadata(df, ledger)

        eligible = self._eligible_for_dft(df, require_mlff=requested_mlff and self.require_mlff_for_dft)
        frontier = pareto_front(
            eligible,
            ["band_score", "stab_score", "transport_score", "dielectric_score", "uncertainty_score"],
            max_input=self.pareto_input_size,
            sort_by="acquisition_score",
            limit=self.frontier_size,
        )
        self._write_frontier(frontier)
        self._update_ledger_after_screen(ledger, df, round_id)

        state = self._load_state()
        state["status"] = "idle"
        state["last_screened_round"] = round_id
        state["last_screened_at"] = _utc()
        state["last_screening"] = {
            "n_candidates": int(len(work_candidates)),
            "n_ranked": int(len(df)),
            "n_mlff": int(n_mlff),
            "n_eligible": int(len(eligible)),
            "n_frontier": int(len(frontier)),
            "mlff_warning": mlff_warning,
        }
        if mlff_warning:
            state["mlff_warning"] = mlff_warning
        else:
            state.pop("mlff_warning", None)
        self._save_state(state)
        return state["last_screening"]

    @staticmethod
    def _replace_rows(base: pd.DataFrame, replacement: pd.DataFrame) -> pd.DataFrame:
        if replacement.empty:
            return base
        keep = base[~base["candidate_id"].astype(str).isin(set(replacement["candidate_id"].astype(str)))]
        return pd.concat([replacement, keep], ignore_index=True, sort=False)

    def _attach_property_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        # `opcional`: `surrogate_energy.pkl` solo lo produce el orquestador de
        # aprendizaje activo (scripts/active_learning_orchestrator.py) y no
        # viaja en el binario. Que falte es normal; que falten los otros no.
        specs = {
            "energy": ("surrogate_energy.pkl", "energy_per_atom_ml_eV", "energy_sigma_eV_atom", True),
            "meff_e": ("surrogate_meff_e.pkl", "meff_e_pred_m0", "meff_e_sigma_m0", False),
            "meff_h": ("surrogate_meff_h.pkl", "meff_h_pred_m0", "meff_h_sigma_m0", False),
            "eps_inf": ("surrogate_eps_inf.pkl", "eps_inf_pred", "eps_inf_sigma", False),
        }
        for _, mean_col, sigma_col, _opt in specs.values():
            out[mean_col] = np.nan
            out[sigma_col] = np.nan

        try:
            from ml_surrogate.features import build_X
            from ml_surrogate.model import SurrogateEnsemble
        except Exception as exc:
            # Antes se devolvía en silencio. Sin estas predicciones, tres de los
            # términos del score caen a su valor neutro y el ranking deja de
            # discriminar — conviene que quede dicho.
            log.warning(
                "sin predicciones de propiedades (%s: %s); transporte, dieléctrico "
                "y excitón usarán su valor por defecto",
                type(exc).__name__, exc,
            )
            return out

        for _, (filename, mean_col, sigma_col, opcional) in specs.items():
            model_path = self._buscar_modelo("models", filename)
            if model_path is None:
                nivel = log.debug if opcional else log.warning
                nivel("modelo de propiedades ausente: %s (columna %s sin poblar)",
                      filename, mean_col)
                continue
            try:
                model = SurrogateEnsemble.load(model_path)
                for col in model.feature_cols:
                    if col not in out:
                        out[col] = np.nan
                X = build_X(out, model.feature_cols)
                means, sigmas = model.predict_batch(X)
            except Exception as exc:
                # Causa habitual: el pickle se serializó con otra versión de
                # scikit-learn. Se nombra, como hace monitor_api.services.ml.
                log.warning("no se pudo usar %s (%s: %s); %s queda sin poblar",
                            model_path.name, type(exc).__name__, exc, mean_col)
                continue
            out[mean_col] = means
            out[sigma_col] = sigmas
        return out

    def _score_objectives(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["band_score"] = pd.to_numeric(out.get("band_score"), errors="coerce").fillna(0.0)
        out["stab_score"] = pd.to_numeric(out.get("stab_score"), errors="coerce").fillna(0.5)
        eg_sigma = pd.to_numeric(out.get("Eg_sigma_eV"), errors="coerce").fillna(0.0)
        out["uncertainty_score"] = eg_sigma.clip(lower=0.0, upper=1.0)

        transport = []
        dielectric = []
        exciton = []
        exciton_mev = []
        # Qué términos salieron de una predicción real y cuáles del valor por
        # defecto. Los tres devuelven 0.5 cuando les falta el dato, y 0.5 no es
        # neutro: es optimista para un material malo y pesimista para uno bueno.
        # Sin esta columna, un score imputado es indistinguible de uno medido.
        imputados = []
        for _, row in out.iterrows():
            m_e, m_h = row.get("meff_e_pred_m0"), row.get("meff_h_pred_m0")
            eps = row.get("eps_inf_pred")
            hay_meff = _finite(m_e) is not None or _finite(m_h) is not None
            hay_eps = _finite(eps) is not None

            transport.append(_transport_score(m_e, m_h))
            dielectric.append(_dielectric_score(eps))
            ex_score, ex_mev = _exciton_score(m_e, m_h, eps)
            exciton.append(ex_score)
            exciton_mev.append(ex_mev)

            faltan = []
            if not hay_meff:
                faltan.append("transporte")
            if not hay_eps:
                faltan.append("dielectrico")
            if ex_mev is None:
                faltan.append("exciton")
            imputados.append(",".join(faltan))

        out["transport_score"] = transport
        out["dielectric_score"] = dielectric
        out["exciton_score"] = exciton
        out["exciton_binding_meV"] = exciton_mev
        out["scores_imputados"] = imputados

        n_imputados = int(sum(1 for v in imputados if v))
        if n_imputados:
            log.warning(
                "%d de %d candidatos puntuados con términos por defecto (%s): "
                "el ranking discrimina con menos información de la que parece",
                n_imputados, len(imputados),
                ", ".join(sorted({t for v in imputados if v for t in v.split(",")})),
            )
        # OJO: los dos `0.5` son marcadores de terminos nunca implementados,
        # con pesos 0.20 y 0.10 -- el 30 % de `pv_score_ml` es una constante.
        # Al ser constante no altera el ORDEN, asi que cambiarlo moveria los
        # numeros publicados sin mejorar el ranking; se deja y se documenta.
        # `dielectric_score` si se calcula, pero solo entra en `acquisition_score`.
        out["pv_score_ml"] = (
            0.25 * out["band_score"]
            + 0.20 * 0.5                      # marcador, sin termino detras
            + 0.20 * out["stab_score"]
            + 0.15 * out["transport_score"]
            + 0.10 * out["exciton_score"]
            + 0.10 * 0.5                      # marcador, sin termino detras
        )
        out["acquisition_score"] = (
            out["pv_score_ml"]
            + 0.10 * out["uncertainty_score"]
            + 0.05 * out["dielectric_score"]
        )
        return out.sort_values("acquisition_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _attach_ledger_metadata(df: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
        if df.empty or ledger.empty:
            return df
        cols = [
            "candidate_id", "A_site", "B_site", "X_site", "B_family",
            "dominant_B", "dominant_halide", "status", "round_selected",
            "round_completed",
        ]
        present = [col for col in cols if col in ledger.columns]
        if present == ["candidate_id"]:
            return df
        return df.merge(
            ledger[present],
            on="candidate_id",
            how="left",
            suffixes=("", "_ledger"),
        )

    def _eligible_for_dft(self, df: pd.DataFrame, *, require_mlff: bool) -> pd.DataFrame:
        if df.empty:
            return df.head(0).copy()
        mask = pd.Series(True, index=df.index)
        if "dropped_at_tier" in df:
            mask &= df["dropped_at_tier"].isna()
        if "passed_eform" in df:
            mask &= df["passed_eform"].fillna(False).astype(bool)
        if "pv_score_ml" in df:
            mask &= pd.to_numeric(df["pv_score_ml"], errors="coerce").fillna(0.0) >= self.min_pv_score
        if require_mlff and "mlff_evaluated" in df:
            mask &= df["mlff_evaluated"].fillna(False).astype(bool)
        return df[mask].copy().reset_index(drop=True)

    def _update_ledger_after_screen(self, ledger: pd.DataFrame, df: pd.DataFrame, round_id: int) -> None:
        if df.empty:
            return
        updates = df.set_index("candidate_id")
        cols = [
            "Eg_surrogate_eV", "Eg_sigma_eV", "Eform_eV_atom", "Eform_std_eV_atom",
            "meff_e_pred_m0", "meff_h_pred_m0", "eps_inf_pred", "exciton_binding_meV",
            "band_score", "stab_score", "transport_score", "dielectric_score",
            "exciton_score", "pv_score_ml", "acquisition_score", "mlff_evaluated",
            "dropped_at_tier", "drop_reason", "passed_eform",
            # La fase no se confirma en el cribado; que el aviso viaje al ledger
            # y de ahi a la frontera y al informe.
            "riesgo_politipo",
            # Que se vea si un score salio de una prediccion o de un valor por defecto.
            "scores_imputados",
        ]
        for col in cols:
            if col not in ledger:
                ledger[col] = "" if col in TEXT_LEDGER_COLUMNS else None
        ledger = self._coerce_ledger_columns(ledger)

        for idx, row in ledger.iterrows():
            cid = row["candidate_id"]
            if cid not in updates.index:
                continue
            scored = updates.loc[cid]
            for col in cols:
                if col in updates.columns:
                    ledger.at[idx, col] = scored.get(col)
            ledger.at[idx, "last_screened_round"] = round_id
            if str(row.get("status", "unseen")) not in DFT_LEDGER_STATUSES:
                eligible = (
                    pd.isna(scored.get("dropped_at_tier"))
                    and bool(scored.get("passed_eform", False))
                    and (_finite(scored.get("pv_score_ml"), 0.0) or 0.0) >= self.min_pv_score
                )
                ledger.at[idx, "status"] = "viable_ml" if eligible else "screened"

        self._write_ledger(ledger)

    def _write_frontier(self, frontier: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frontier.to_csv(self.frontier_path, index=False)

    # ── Rounds ─────────────────────────────────────────────────────────────

    def advance(
        self,
        *,
        start_runner: bool = True,
        dry_run: bool = False,
        use_mlff: bool | None = None,
    ) -> dict[str, Any]:
        """Advance the discovery loop by one state transition."""
        if not self.ledger_path.exists():
            self.init_space()

        state = self._load_state()
        if state.get("status") == "paused":
            return self.status()
        if state.get("status") == "dft_selected":
            if dry_run:
                return self.status()
            round_id = int(state.get("active_round", state.get("current_round", 0)))
            candidate_ids = self._round_candidate_ids(round_id)
            if not candidate_ids:
                state["status"] = "idle"
                state["stop_reason"] = "ronda preseleccionada sin candidatos recuperables"
                self._save_state(state)
                return self.status()
            prepared = self.prepare_round(
                round_id,
                candidate_ids,
                start_runner=start_runner,
                dry_run=False,
            )
            state = self._load_state()
            state["last_prepared"] = prepared
            self._save_state(state)
            return self.status()
        if state.get("status") in {"dft_prepared", "dft_running"}:
            round_id = int(state.get("current_round", 0))
            if not self._round_finished(round_id):
                diagnostics = self.runner_diagnostics(round_id, state=state)
                if diagnostics.get("stale"):
                    # Un runner que se cae a medias deja jobs en "running" con un
                    # PID muerto; sin devolverlos a "pending" el round nunca
                    # cuenta como terminado y el bucle se cuelga para siempre.
                    n_reset = self._reset_phantom_running(round_id)

                    # Cortafuegos: si tras varios relanzamientos SEGUIDOS nada
                    # progresa, WSL está roto de verdad — mejor un estado de
                    # error legible que un bucle relanzando cada 10 min sin fin.
                    # Si el relanzamiento anterior sí completó algún job, el
                    # incidente se considera nuevo y el contador vuelve a empezar.
                    finished_now = (
                        (diagnostics.get("status_counts") or {}).get("converged", 0)
                        + (diagnostics.get("status_counts") or {}).get("dft_failed", 0)
                        + (diagnostics.get("status_counts") or {}).get("failed", 0)
                    )
                    if finished_now > int(state.get("stale_relaunch_finished", -1)):
                        intentos = 1
                    else:
                        intentos = int(state.get("stale_relaunches", 0)) + 1
                    max_intentos = int(self.discovery.get("max_stale_relaunches", 5))
                    if intentos > max_intentos:
                        state["status"] = "error"
                        state["last_error"] = (
                            f"El runner DFT de la ronda {round_id} no progresa tras "
                            f"{max_intentos} relanzamientos. Revisa el runtime WSL/GPAW "
                            f"y reanuda con /api/discovery/run."
                        )
                        state.pop("stale_relaunches", None)
                        self._save_state(state)
                        return self.status()

                    state["status"] = "dft_prepared"
                    state["stale_relaunches"] = intentos
                    state["stale_relaunch_finished"] = finished_now
                    if diagnostics.get("error"):
                        state["runner_error"] = diagnostics["error"]
                    elif diagnostics.get("no_progress"):
                        state["runner_error"] = (
                            f"sin progreso del runner en {diagnostics.get('progress_age_sec')} s"
                            + (f"; {n_reset} job(s) 'running' fantasma devueltos a 'pending'"
                               if n_reset else "")
                        )
                    self._set_round_ledger_status(
                        round_id,
                        "dft_prepared",
                        only={"dft_running", "dft_selected"},
                    )
                    self._save_state(state)
                    if not start_runner:
                        return self.status()
                if start_runner and state.get("status") == "dft_prepared":
                    try:
                        runner_info = self._launch_runner(self._round_runs_dir(round_id))
                    except Exception as exc:
                        state["status"] = "dft_prepared"
                        state["runner_error"] = f"{type(exc).__name__}: {exc}"
                        self._set_round_ledger_status(
                            round_id,
                            "dft_prepared",
                            only={"dft_running", "dft_selected"},
                        )
                        self._save_state(state)
                        raise
                    state["status"] = "dft_running"
                    state["runner_started_at"] = _utc()
                    state["runner"] = runner_info
                    state.pop("runner_error", None)
                    self._save_state(state)
                return self.status()
            self.finalize_round(round_id, retrain=True)
            return self.status()

        screening = self.score_space(use_mlff=use_mlff)
        selected = self.select_next_batch()
        if not selected:
            state = self._load_state()
            state["status"] = "done"
            state["stop_reason"] = "no quedan candidatos top viables sin verificar por DFT"
            self._save_state(state)
            return self.status()

        state = self._load_state()
        round_id = int(state.get("current_round", 0))
        prepared = self.prepare_round(round_id, selected, start_runner=start_runner, dry_run=dry_run)
        state = self._load_state()
        state["last_screening"] = screening
        state["last_prepared"] = prepared
        self._save_state(state)
        return self.status()

    def select_next_batch(self) -> list[str]:
        frontier = pd.read_csv(self.frontier_path) if self.frontier_path.is_file() else pd.DataFrame()
        ledger = self._read_ledger()
        if frontier.empty or ledger.empty:
            return []
        used = set(ledger[ledger["status"].isin(DFT_LEDGER_STATUSES)]["candidate_id"].astype(str))
        eligible = frontier[~frontier["candidate_id"].astype(str).isin(used)].copy()
        if eligible.empty:
            return []
        return self._diversified_select(eligible, self.batch_size)

    def _diversified_select(self, df: pd.DataFrame, n: int) -> list[str]:
        work = df.sort_values("acquisition_score", ascending=False).copy()
        max_b = max(1, int(self.discovery.get("max_per_b_family", math.ceil(n / 2))))
        max_x = max(1, int(self.discovery.get("max_per_halide", math.ceil(n * 0.6))))
        selected: list[str] = []
        counts_b: dict[str, int] = {}
        counts_x: dict[str, int] = {}

        def take(row) -> None:
            cid = str(row["candidate_id"])
            selected.append(cid)
            b = str(row.get("B_family") or row.get("dominant_B") or "")
            x = str(row.get("dominant_halide") or "")
            counts_b[b] = counts_b.get(b, 0) + 1
            counts_x[x] = counts_x.get(x, 0) + 1

        for _, row in work.iterrows():
            if len(selected) >= n:
                break
            b = str(row.get("B_family") or row.get("dominant_B") or "")
            x = str(row.get("dominant_halide") or "")
            if counts_b.get(b, 0) >= max_b or counts_x.get(x, 0) >= max_x:
                continue
            take(row)

        if len(selected) < n:
            for _, row in work.iterrows():
                cid = str(row["candidate_id"])
                if cid not in selected:
                    take(row)
                if len(selected) >= n:
                    break
        return selected

    def _dedup_estructural(
        self,
        candidate_ids: list[str],
        candidates: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Quita los candidatos que construyen una estructura ya calculada.

        La formula no sirve como identidad: `Cs0.95Rb0.051SnI3` y `CsSnI3` se
        escriben distinto y producen los mismos atomos, porque 8 sitios A no
        pueden alojar un 5 % de Rb. Mandar los dos a DFT cuesta el doble para
        obtener el mismo numero, y ademas mete duplicados en el entrenamiento:
        de 111 filas del CSV, 56 compartian Eg = 1.075 eV con 48 formulas
        distintas, lo que hacia parecer que el surrogate generalizaba cuando
        solo reconocia repetidos repartidos entre folds de la validacion.
        """
        supercell = (self.config.get("structure", {}) or {}).get(
            "supercell_mixed", [2, 2, 2]
        )

        # Estructuras que ya tienen (o van a tener) su DFT.
        vistas: dict[tuple, str] = {}
        ledger = self._read_ledger()
        if not ledger.empty and "status" in ledger.columns:
            ya_corridos = ledger[ledger["status"].isin(DFT_LEDGER_STATUSES)]
            for cid in ya_corridos["candidate_id"].astype(str):
                cand = candidates.get(cid)
                if cand is not None:
                    vistas.setdefault(clave_estructura(cand, supercell), cid)

        conservados: list[str] = []
        descartados: list[dict[str, str]] = []
        for cid in candidate_ids:
            cand = candidates.get(cid)
            if cand is None:
                conservados.append(cid)
                continue
            clave = clave_estructura(cand, supercell)
            duplicado_de = vistas.get(clave)
            if duplicado_de is None:
                vistas[clave] = cid
                conservados.append(cid)
            else:
                descartados.append({
                    "candidate_id": cid,
                    "formula": getattr(cand, "formula", ""),
                    "duplica_a": duplicado_de,
                })

        if descartados:
            log.warning(
                "ronda: %d de %d candidatos construyen una estructura ya "
                "calculada y no se mandan a DFT (p.ej. %s duplica a %s)",
                len(descartados), len(candidate_ids),
                descartados[0]["formula"], descartados[0]["duplica_a"],
            )
        return conservados, descartados

    def prepare_round(
        self,
        round_id: int,
        candidate_ids: list[str],
        *,
        start_runner: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        candidates = self._load_candidates()
        candidate_ids, descartados = self._dedup_estructural(candidate_ids, candidates)
        selected = [candidates[cid] for cid in candidate_ids if cid in candidates]
        round_dir = self._round_dir(round_id)
        round_dir.mkdir(parents=True, exist_ok=True)

        ledger = self._read_ledger()
        selected_df = ledger[ledger["candidate_id"].astype(str).isin(candidate_ids)].copy()
        selected_df.to_csv(round_dir / "selected_for_dft.csv", index=False)

        status = "dft_selected" if dry_run else "dft_prepared"
        ledger.loc[ledger["candidate_id"].astype(str).isin(candidate_ids), "status"] = status
        ledger.loc[ledger["candidate_id"].astype(str).isin(candidate_ids), "round_selected"] = round_id
        self._write_ledger(ledger)

        manifest = {
            "round_id": round_id,
            "selected_at": _utc(),
            "n_selected": len(selected),
            "candidate_ids": candidate_ids,
            # Los que iban a repetir una estructura ya calculada. Se registran
            # en vez de desaparecer: si la ronda sale mas corta de lo pedido,
            # aqui esta el porque.
            "descartados_por_duplicado": descartados,
            "dry_run": dry_run,
            "status": status,
        }

        n_prepared = 0
        if not dry_run:
            from buho.dft_jobs.prepare_relaxation_jobs import RelaxationJobPreparer

            runs_dir = self._round_runs_dir(round_id)
            preparer = RelaxationJobPreparer(
                self.config,
                project_root=self.project_root,
                n_cores=self.runner_cores,
            )
            prepared = preparer.prepare(selected, out_root=runs_dir, config_src=self.config_path)
            self._mark_discovery_jobs(runs_dir, round_id, candidate_ids)
            n_prepared = len(prepared)
            manifest.update(
                status="dft_prepared",
                runs_dir=str(runs_dir),
                n_prepared=n_prepared,
            )
            if start_runner and selected:
                runner_error = None
                try:
                    runner_info = self._launch_runner(runs_dir)
                except Exception as exc:
                    runner_error = f"{type(exc).__name__}: {exc}"
                    manifest["runner_error"] = runner_error
                    manifest["status"] = "dft_prepared"
                else:
                    manifest["status"] = "dft_running"
                    manifest["runner"] = runner_info
                    ledger = self._read_ledger()
                    ledger.loc[ledger["candidate_id"].astype(str).isin(candidate_ids), "status"] = "dft_running"
                    self._write_ledger(ledger)

        (round_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        state = self._load_state()
        state["status"] = manifest["status"]
        state["current_round"] = round_id
        state["active_round"] = round_id
        state["active_round_dir"] = str(round_dir)
        state["active_runs_dir"] = str(self._round_runs_dir(round_id))
        state["n_selected_active"] = len(selected)
        if manifest.get("runner_error"):
            state["runner_error"] = manifest["runner_error"]
        else:
            state.pop("runner_error", None)
        if manifest.get("runner"):
            state["runner"] = manifest["runner"]
        else:
            state.pop("runner", None)
        self._save_state(state)
        if manifest.get("runner_error"):
            raise RuntimeError(str(manifest["runner_error"]))
        return {"round_id": round_id, "n_selected": len(selected), "n_prepared": n_prepared}

    def _round_candidate_ids(self, round_id: int) -> list[str]:
        round_dir = self._round_dir(round_id)
        manifest_path = round_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
            ids = manifest.get("candidate_ids")
            if isinstance(ids, list):
                return [str(cid) for cid in ids if cid]

        selected = round_dir / "selected_for_dft.csv"
        if selected.is_file():
            try:
                df = pd.read_csv(selected)
            except pd.errors.EmptyDataError:
                return []
            if "candidate_id" in df:
                return [str(cid) for cid in df["candidate_id"].dropna().tolist()]
        return []

    def _set_round_ledger_status(
        self,
        round_id: int,
        status: str,
        *,
        only: set[str] | None = None,
    ) -> None:
        candidate_ids = self._round_candidate_ids(round_id)
        if not candidate_ids:
            return
        ledger = self._read_ledger()
        if ledger.empty or "candidate_id" not in ledger or "status" not in ledger:
            return
        mask = ledger["candidate_id"].astype(str).isin(set(candidate_ids))
        if only:
            mask &= ledger["status"].astype(str).isin(only)
        if not mask.any():
            return
        ledger.loc[mask, "status"] = status
        self._write_ledger(ledger)

    def _launch_runner(self, runs_dir: Path) -> dict[str, Any]:
        script = self.project_root / "scripts" / "buho_relax_runner.py"
        if not script.is_file():
            raise FileNotFoundError(f"No se encuentra el runner DFT: {script}")
        runs_dir.mkdir(parents=True, exist_ok=True)
        backend, cmd, cwd = self._runner_command(runs_dir, preflight_only=False)
        _, preflight_cmd, preflight_cwd = self._runner_command(runs_dir, preflight_only=True)
        command_record: dict[str, Any] = {
            "backend": backend,
            "cwd": cwd,
            "command": cmd,
            "preflight_command": preflight_cmd,
            "started_at": _utc(),
        }
        (runs_dir / "runner_command.json").write_text(
            json.dumps(command_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                preflight_cmd,
                cwd=preflight_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.runner_preflight_timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            message = f"No se pudo ejecutar el preflight DFT: {exc}"
            with (runs_dir / "runner.out").open("a", encoding="utf-8") as out:
                out.write(f"\n===== DFT runner preflight {datetime.now().isoformat(timespec='seconds')} =====\n")
                out.write(f"backend={backend}\n")
                out.write(message + "\n")
            raise RuntimeError(message) from exc
        preflight_output = (result.stdout or "").strip()
        with (runs_dir / "runner.out").open("a", encoding="utf-8") as out:
            out.write(f"\n===== DFT runner preflight {datetime.now().isoformat(timespec='seconds')} =====\n")
            out.write(f"backend={backend}\n")
            out.write(" ".join(cmd) + "\n")
            if preflight_output:
                out.write(preflight_output + "\n")
            out.write(f"preflight_returncode={result.returncode}\n")
        if result.returncode != 0:
            error = self._runner_error_from_text(preflight_output) or preflight_output.splitlines()[-1:]
            if isinstance(error, list):
                error = error[0] if error else f"preflight_returncode={result.returncode}"
            raise RuntimeError(str(error))

        out = open(runs_dir / "runner.out", "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=out,
            stderr=subprocess.STDOUT,
            **self._detached_kwargs(),
        )
        command_record["pid"] = proc.pid
        command_record["runner_started_at"] = _utc()
        (runs_dir / "runner_command.json").write_text(
            json.dumps(command_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return command_record

    @staticmethod
    def _mark_discovery_jobs(runs_dir: Path, round_id: int, candidate_ids: list[str]) -> None:
        selected_at = _utc()
        for cid in candidate_ids:
            meta_path = runs_dir / cid / "metadata.json"
            if not meta_path.is_file():
                continue
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            data.update({
                "discovery_loop": True,
                "discovery_round": round_id,
                "discovery_selected_at": selected_at,
            })
            meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _round_finished(self, round_id: int) -> bool:
        runs_dir = self._round_runs_dir(round_id)
        statuses = list(runs_dir.glob("*/status.json"))
        if not statuses:
            return False
        for path in statuses:
            try:
                state = json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")
            except json.JSONDecodeError:
                return False
            if state in ACTIVE_JOB_STATUSES:
                return False
        return True

    def finalize_round(self, round_id: int, *, retrain: bool = True) -> dict[str, Any]:
        from buho.dft_jobs.collect_results import ResultCollector

        round_dir = self._round_dir(round_id)
        runs_dir = self._round_runs_dir(round_id)
        collector = ResultCollector(project_root=self.project_root)
        results = collector.collect_all(runs_dir)
        round_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(round_dir / "dft_results.csv", index=False)

        ledger = self._read_ledger()
        if not results.empty:
            by_id = results.set_index("candidate_id")
            result_cols = [
                "converged", "energy_per_atom_eV", "bandgap_preliminary_eV",
                "final_energy_eV", "trusted_label", "error_message", "calc_time_s",
            ]
            for col in result_cols:
                if col not in ledger:
                    ledger[col] = pd.NA if col in TEXT_LEDGER_COLUMNS | BOOLEAN_LEDGER_COLUMNS else np.nan
            ledger = self._coerce_ledger_columns(ledger)
            for idx, row in ledger.iterrows():
                cid = row["candidate_id"]
                if cid not in by_id.index:
                    continue
                res = by_id.loc[cid]
                converged = bool(res.get("converged", False))
                trusted = bool(res.get("trusted_label", False))
                ledger.at[idx, "status"] = "dft_converged" if converged else "dft_failed"
                ledger.at[idx, "round_completed"] = round_id
                for col in result_cols:
                    if col in by_id.columns:
                        ledger.at[idx, col] = res.get(col)
                if converged and trusted:
                    ledger.at[idx, "status"] = "training_added"

        appended = self._append_training(results, ledger, round_id)
        self._write_ledger(ledger)
        metrics = self._retrain_bandgap(round_id) if retrain and appended else {"status": "skipped"}

        manifest = {
            "round_id": round_id,
            "finalized_at": _utc(),
            "n_results": int(len(results)),
            "n_converged": int(results["converged"].sum()) if "converged" in results else 0,
            "n_training_appended": int(appended),
            "retrain": metrics,
            "status": "complete",
        }
        (round_dir / "manifest.final.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        state = self._load_state()
        state["status"] = "idle"
        state["current_round"] = round_id + 1
        state["last_completed_round"] = round_id
        state["last_finalized_at"] = _utc()
        state["last_finalize"] = manifest
        # La ronda terminó: el contador de relanzamientos por atasco no debe
        # arrastrarse a la siguiente.
        state.pop("stale_relaunches", None)
        state.pop("stale_relaunch_finished", None)
        state.pop("runner_error", None)
        self._save_state(state)
        return manifest

    def _reset_phantom_running(self, round_id: int) -> int:
        """Devuelve a "pending" los jobs que un runner muerto dejó en "running".

        `_round_finished` trata "running" como activo, así que un job huérfano
        —su runner ya no existe— impide que la ronda cuente como terminada y el
        bucle se queda esperando indefinidamente. Marcarlo "pending" deja que el
        siguiente runner lo reintente.
        """
        runs_dir = self._round_runs_dir(round_id)
        reset = 0
        for path in runs_dir.glob("*/status.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("status") not in {"running", "stalled", "oscillating"}:
                continue
            payload["status"] = "pending"
            payload["phantom_reset_at"] = _utc()
            payload.pop("pid", None)
            try:
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                reset += 1
            except OSError:
                pass
        return reset

    def _append_training(self, results: pd.DataFrame, ledger: pd.DataFrame, round_id: int) -> int:
        if results.empty or "trusted_label" not in results:
            return 0
        trusted = results[
            (results.get("trusted_label") == True)  # noqa: E712
            & pd.to_numeric(results.get("bandgap_preliminary_eV"), errors="coerce").notna()
        ].copy()
        if trusted.empty:
            return 0

        feature_cols = [
            "candidate_id", "formula", "tolerance_t", "oct_factor", "vol_est_A3",
            "Eform_eV_atom", "energy_per_atom_eV", "bandgap_preliminary_eV",
            "Eg_surrogate_eV", "Eg_sigma_eV",
        ]
        led = ledger[[c for c in feature_cols if c in ledger.columns]].copy()
        merged = trusted.merge(led, on="candidate_id", how="left", suffixes=("", "_ledger"))

        candidatos = self._load_candidates()
        tabla_soc = bandgap_scissor.cargar_tabla()

        rows = []
        for _, row in merged.iterrows():
            eg = _finite(row.get("bandgap_preliminary_eV"))
            if eg is None:
                continue
            # El gap de PBE del cribado no lleva acoplamiento espín-órbita, que
            # es un efecto grande y dependiente del elemento B. Se corrige antes
            # de que el número se convierta en etiqueta de entrenamiento: si se
            # corrigiera después, el surrogate ya habría aprendido el sesgo.
            # `band_gap_gga_eV` conserva el valor crudo para poder auditar.
            cand = candidatos.get(str(row.get("candidate_id")))
            fracciones_b = cand.fractions.get("B", {}) if cand is not None else {}
            eg_corregido = bandgap_scissor.corregir(eg, fracciones_b, tabla_soc)
            rows.append({
                "material_id": row.get("candidate_id"),
                "candidate_id": row.get("candidate_id"),
                "formula": row.get("formula") or row.get("formula_ledger"),
                "tolerance_t": row.get("tolerance_t"),
                "oct_factor": row.get("oct_factor"),
                "vol_est_A3": row.get("vol_est_A3"),
                "Eform_eV_atom": row.get("Eform_eV_atom"),
                "band_gap_gga_eV": eg,
                "energy_per_atom_eV": row.get("energy_per_atom_eV"),
                "chi_soc_eV": round(eg_corregido - eg, 5),
                "Eg_target_eV": eg_corregido,
                "split": _split_for_candidate(str(row.get("candidate_id"))),
                "source": f"discovery_round_{round_id:03d}",
                "added_at": _utc(),
            })
        if not rows:
            return 0

        new = pd.DataFrame(rows)
        if self.training_path.exists():
            old = pd.read_csv(self.training_path)
            new = pd.concat([old, new], ignore_index=True)
            new = new.drop_duplicates(subset=["candidate_id", "source"], keep="last")
        self.training_path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(self.training_path, index=False)
        return len(rows)

    def _sin_estructuras_repetidas(self, df: "pd.DataFrame") -> tuple["pd.DataFrame", int]:
        """Una fila por estructura antes de entrenar y validar.

        La validacion cruzada reparte filas al azar entre folds. Si la misma
        estructura aparece varias veces con formulas distintas —lo que pasaba
        cuando el generador proponia dopajes que la supercelda no puede
        representar— acaba a ambos lados del corte y el `cv_mae` mide
        reconocimiento de repetidos, no generalizacion.

        Medido sobre el CSV real: 111 filas eran 21 estructuras, y el `cv_mae`
        pasaba de 0.0036 eV a 0.0330 eV al deduplicar, contra un baseline de
        0.0363. Es la diferencia entre "veinte veces mejor que la media" y
        "un 9 % mejor".
        """
        if "candidate_id" not in df.columns or df.empty:
            return df, 0
        supercell = (self.config.get("structure", {}) or {}).get("supercell_mixed", [2, 2, 2])
        candidatos = self._load_candidates()

        claves: list[str | None] = []
        for cid in df["candidate_id"].astype(str):
            cand = candidatos.get(cid)
            claves.append(str(clave_estructura(cand, supercell)) if cand is not None else None)

        trabajo = df.copy()
        trabajo["_clave"] = claves
        # Las filas cuyo candidato no se localiza se conservan: no se puede
        # afirmar que dupliquen nada.
        sin_clave = trabajo[trabajo["_clave"].isna()]
        con_clave = trabajo[trabajo["_clave"].notna()].drop_duplicates(subset=["_clave"])
        resultado = pd.concat([con_clave, sin_clave], ignore_index=True).drop(columns=["_clave"])
        return resultado, int(len(df) - len(resultado))

    def _retrain_bandgap(self, round_id: int) -> dict[str, Any]:
        try:
            from ml_surrogate.features import BASE_FEATURES, build_X
            from ml_surrogate.model import SurrogateEnsemble
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        frames = []
        lit_path = self.data_root / "data" / "surrogate_training.csv"
        if lit_path.is_file():
            frames.append(pd.read_csv(lit_path))
        if self.training_path.is_file():
            frames.append(pd.read_csv(self.training_path))
        if not frames:
            return {"status": "skipped", "reason": "sin datos de entrenamiento"}

        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["Eg_target_eV"])
        df, n_duplicadas = self._sin_estructuras_repetidas(df)
        feat_cols = [col for col in BASE_FEATURES if col in df.columns]
        # `band_gap_gga_eV` NO entra: en este bucle el target es el bandgap del
        # propio DFT, y esa columna guarda ese mismo número (Eg_target_eV es
        # Eg_pbe + chi_SOC, y chi_SOC depende solo del sitio B). Usarla como
        # feature es predecir el target desde el target: daba cv_mae 0.002 eV,
        # veinte veces mejor que la media, mientras el modelo solo copiaba una
        # columna. Y en produccion no existe —es justo lo que hay que predecir
        # antes del DFT—, asi que `build_X` la rellenaba con 0.0 frente a un
        # valor tipico de 1.05: cada prediccion real extrapolaba fuera de rango.
        for col in ("a_lat_mp_A", "Eform_eV_atom"):
            if col in df.columns and col not in feat_cols:
                feat_cols.append(col)
        if len(df) < 5 or len(feat_cols) < 4:
            return {"status": "skipped", "reason": "datos insuficientes", "n_samples": int(len(df))}

        X = build_X(df, feat_cols)
        y = df["Eg_target_eV"].values.astype(float)
        model = SurrogateEnsemble().fit(X, y, feat_cols)
        # Se sella con la escala de las etiquetas. Sin el sello, la cascada no
        # sabe si puede llevar la prediccion a escala experimental, y aplicar
        # una correccion calibrada para otra escala da un numero sin sentido
        # que no se nota mirando el resultado.
        eg_scale.sellar_modelo(model)

        model_dir = self.models_root / "models" / "discovery"
        model_dir.mkdir(parents=True, exist_ok=True)
        version_path = model_dir / f"surrogate_bandgap_round_{round_id:03d}.pkl"
        model.save(version_path)

        # Publicar el modelo donde la cascada lo lee. Sin esto el bucle
        # reentrenaba cada ronda, guardaba el .pkl versionado y seguía cribando
        # con el modelo de fábrica: proponía la misma química una y otra vez
        # mientras el DFT la desmentía, y el aprendizaje activo no cerraba.
        # La copia se hace a un temporal y se renombra para que nadie lea un
        # fichero a medio escribir.
        current_path = model_dir / "surrogate_bandgap_current.pkl"
        try:
            tmp_path = current_path.with_suffix(".pkl.tmp")
            shutil.copyfile(version_path, tmp_path)
            os.replace(tmp_path, current_path)
        except OSError as exc:
            # Que no se pueda publicar no invalida la ronda, pero hay que
            # decirlo: significa que la siguiente criba usará el modelo viejo.
            log.warning("no se pudo publicar el surrogate en %s: %s", current_path, exc)
            current_path = None

        preds, _ = model.predict_batch(X)
        train_mae = float(np.mean(np.abs(preds - y)))
        rec = {
            "round_id": round_id,
            "status": "ok",
            "n_samples": int(len(df)),
            # Filas descartadas por describir una estructura ya presente. Si es
            # alto, el DFT se esta gastando en repetir material.
            "n_duplicadas_descartadas": int(n_duplicadas),
            "n_features": int(len(feat_cols)),
            # Error de AJUSTE, no de generalización: se predice sobre las mismas
            # filas con las que se entrenó. Se conserva porque delata problemas
            # (si sube mucho, el modelo ni siquiera ajusta), pero no mide
            # calidad: ver cv_mae_eV.
            "train_mae_eV": round(train_mae, 5),
            **_cv_metrics(X, y, feat_cols),
            "model_path": str(version_path),
            # Qué modelo usará la siguiente criba. Si sale None, el bucle
            # seguirá cribando con el anterior y hay que mirarlo.
            "published_to": str(current_path) if current_path else None,
            "trained_at": _utc(),
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ── Loop controls ──────────────────────────────────────────────────────

    def run_forever(
        self,
        *,
        start_runner: bool = True,
        dry_run: bool = False,
        use_mlff: bool | None = None,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        rounds_started = 0
        while True:
            before = self._load_state()
            status = self.advance(start_runner=start_runner, dry_run=dry_run, use_mlff=use_mlff)
            after = self._load_state()
            if after.get("status") in {"done", "paused", "error", "dft_selected"}:
                return status
            if dry_run:
                return status
            if after.get("current_round") != before.get("current_round"):
                rounds_started += 1
            if max_rounds is not None and rounds_started >= max_rounds:
                return status
            if after.get("status") in {"dft_prepared", "dft_running"}:
                time.sleep(self.poll_interval_sec)
                continue
            time.sleep(self.poll_interval_sec)

    def pause(self) -> dict[str, Any]:
        state = self._load_state()
        if state.get("status") not in {"done", "not_initialized"}:
            state["previous_status"] = state.get("status")
            state["status"] = "paused"
            self._save_state(state)
        return self.status()

    def resume(self) -> dict[str, Any]:
        state = self._load_state()
        if state.get("status") == "paused":
            state["status"] = state.get("previous_status") or "idle"
            state.pop("previous_status", None)
            self._save_state(state)
        return self.status()

    # ── Reporting ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        state = self._load_state()
        ledger = self._read_ledger()
        counts = {}
        if not ledger.empty and "status" in ledger:
            counts = {str(k): int(v) for k, v in ledger["status"].value_counts().to_dict().items()}
        total = int(len(ledger))
        seen = total - counts.get("unseen", 0)
        frontier = self.frontier(limit=20)
        active_round = state.get("active_round", state.get("current_round", 0))
        queue = self.round_queue(int(active_round), limit=30) if total else []
        runner = self.runner_diagnostics(int(active_round), state=state) if total else {}
        return {
            "state": state,
            "counts": counts,
            "coverage": {
                "total": total,
                "seen": int(seen),
                "percent": round((seen / total) * 100.0, 2) if total else 0.0,
            },
            "frontier": frontier,
            "queue": queue,
            "paths": self._paths_payload(),
            "config": self._config_payload(),
            "runner": runner,
        }

    def runner_diagnostics(self, round_id: int, *, state: dict[str, Any] | None = None) -> dict[str, Any]:
        runs_dir = self._round_runs_dir(round_id)
        if not runs_dir.is_dir():
            return {"round_id": round_id, "exists": False, "runs_dir": str(runs_dir)}

        status_counts: dict[str, int] = {}
        running_pids: list[int] = []
        for status_path in runs_dir.glob("*/status.json"):
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = "unknown"
                pid = None
            else:
                status = str(payload.get("status") or "unknown")
                pid = payload.get("pid")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "running" and isinstance(pid, int):
                running_pids.append(pid)

        runner_out = runs_dir / "runner.out"
        runner_log = runs_dir / "runner.log"
        runner_command_path = runs_dir / "runner_command.json"
        try:
            runner_command = json.loads(runner_command_path.read_text(encoding="utf-8"))
        except Exception:
            runner_command = {}
        runner_out_tail = self._tail_text(runner_out)
        runner_log_tail = self._tail_text(runner_log)
        error_text = self._runner_error_from_text(runner_out_tail) or self._runner_error_from_text(runner_log_tail)
        has_gpaw_logs = any(runs_dir.glob("*/r2scan.txt")) or any(runs_dir.glob("*/gpaw*.log"))
        total = sum(status_counts.values())
        active = status_counts.get("running", 0) > 0
        all_pending = total > 0 and status_counts.get("pending", 0) == total
        finished = status_counts.get("converged", 0) + status_counts.get("dft_failed", 0) + status_counts.get("failed", 0)
        unfinished = total - finished
        expected_running = (state or {}).get("status") in {"dft_prepared", "dft_running"}

        # Antigüedad de la última señal de vida del runner. Un runner sano escribe
        # una línea STATUS cada ~30 s y los GPAW logs crecen sin parar; si nada de
        # esto se ha tocado en `runner_stale_after_sec`, el runner murió — aunque
        # queden jobs en "running" con un PID que ya no existe (el bug que dejaba
        # el bucle colgado horas: el runner WSL se caía tras completar 15/30 y la
        # detección de "stale" solo miraba si TODOS los jobs seguían pendientes).
        progreso_files = [runner_out, runner_log, *runs_dir.glob("*/status.json")]
        progreso_files += list(runs_dir.glob("*/*.log")) + list(runs_dir.glob("*/*.txt"))
        mtimes = []
        for p in progreso_files:
            try:
                mtimes.append(p.stat().st_mtime)
            except OSError:
                pass
        progress_age = (time.time() - max(mtimes)) if mtimes else None
        stale_after = int(self.discovery.get("runner_stale_after_sec", 600))
        no_progress = progress_age is not None and progress_age > stale_after

        stale = bool(
            expected_running
            and total > 0
            and unfinished > 0
            and (error_text or runner_out.is_file())
            and (
                (all_pending and not active)   # el runner murió sin empezar
                or no_progress                 # el runner murió a medias: nada avanza
            )
        )
        return {
            "round_id": round_id,
            "exists": True,
            "runs_dir": str(runs_dir),
            "status_counts": status_counts,
            "running_pids": running_pids,
            "has_runner_out": runner_out.is_file(),
            "has_runner_log": runner_log.is_file(),
            "has_runner_command": runner_command_path.is_file(),
            "runner_command": runner_command,
            "has_gpaw_logs": bool(has_gpaw_logs),
            "stale": stale,
            "progress_age_sec": round(progress_age, 1) if progress_age is not None else None,
            "no_progress": no_progress,
            "unfinished": unfinished,
            "error": error_text,
            "runner_out_tail": runner_out_tail,
            "runner_log_tail": runner_log_tail,
        }

    @staticmethod
    def _tail_text(path: Path, *, max_chars: int = 1200) -> str:
        if not path.is_file():
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        return text[-max_chars:].strip()

    @staticmethod
    def _runner_error_from_text(text: str) -> str | None:
        if not text:
            return None
        known = [
            "No se encuentran los datasets PAW de GPAW",
            "GPAW_SETUP_PATH no contiene",
            "No se encontro",
            "No se pudo ejecutar el preflight DFT",
            "Preflight DFT fallo",
            "No module named",
            "bad interpreter",
            "No such file or directory",
            "command not found",
            "El runner DFT usa",
        ]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            for needle in known:
                if needle in line:
                    return line
        if "Traceback" in text and lines:
            return lines[-1]
        return None

    def frontier(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not self.frontier_path.is_file():
            return []
        try:
            df = pd.read_csv(self.frontier_path)
        except pd.errors.EmptyDataError:
            return []
        if df.empty:
            return []
        df = self._merge_ledger_view(df)
        return self._records(df.head(limit))

    def round_queue(self, round_id: int, *, limit: int = 30) -> list[dict[str, Any]]:
        selected = self._round_dir(round_id) / "selected_for_dft.csv"
        if not selected.is_file():
            return []
        try:
            df = pd.read_csv(selected)
        except pd.errors.EmptyDataError:
            return []
        df = self._merge_ledger_view(df)
        return self._records(df.head(limit))

    def _merge_ledger_view(self, df: pd.DataFrame) -> pd.DataFrame:
        ledger = self._read_ledger()
        if df.empty or ledger.empty or "candidate_id" not in df or "candidate_id" not in ledger:
            return df
        cols = [
            "candidate_id",
            "status",
            "round_selected",
            "round_completed",
            "last_screened_round",
            "drop_reason",
        ]
        meta_cols = [col for col in cols if col in ledger]
        if len(meta_cols) <= 1:
            return df
        drop_cols = [col for col in meta_cols if col != "candidate_id" and col in df]
        return df.drop(columns=drop_cols, errors="ignore").merge(
            ledger[meta_cols],
            on="candidate_id",
            how="left",
        )

    @staticmethod
    def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        for row in df.to_dict(orient="records"):
            out.append({key: _json_safe(value) for key, value in row.items()})
        return out

    def export(self) -> dict[str, Any]:
        status = self.status()
        report_path = self.output_dir / "discovery_report.md"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# PEROVOWL Discovery Loop",
            "",
            f"- Estado: {status['state'].get('status')}",
            f"- Ronda actual: {status['state'].get('current_round')}",
            f"- Cobertura: {status['coverage']['seen']} / {status['coverage']['total']} ({status['coverage']['percent']}%)",
            f"- Ledger: `{self.ledger_path}`",
            f"- Frontera Pareto: `{self.frontier_path}`",
            "",
            "## Top frontera Pareto",
            "",
            "| Fórmula | Eg ML | Eform | PV ML | adquisición | estado | fase |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        n_riesgo = 0
        for item in status["frontier"][:30]:
            riesgo = item.get("riesgo_politipo")
            if riesgo:
                n_riesgo += 1
            lines.append(
                "| {formula} | {eg} | {eform} | {pv} | {acq} | {st} | {fase} |".format(
                    formula=item.get("formula") or "",
                    eg=item.get("Eg_surrogate_eV") if item.get("Eg_surrogate_eV") is not None else "",
                    eform=item.get("Eform_eV_atom") if item.get("Eform_eV_atom") is not None else "",
                    pv=item.get("pv_score_ml") if item.get("pv_score_ml") is not None else "",
                    acq=item.get("acquisition_score") if item.get("acquisition_score") is not None else "",
                    st=item.get("status") or "",
                    fase="marginal" if riesgo else "plausible",
                )
            )

        # El cribado no confirma en qué fase cristaliza un material: el factor de
        # tolerancia dice si los iones empaquetan en una perovskita cúbica, no si
        # esa es la fase que gana a temperatura ambiente. Decirlo aquí, donde se
        # leen los resultados, y no solo en la documentación.
        lines += [
            "",
            "## Sobre la fase",
            "",
            "Ninguna fila de esta tabla tiene la fase confirmada. El cribado",
            "evalúa la perovskita cúbica ideal; que sea la fase estable a",
            "temperatura ambiente es una hipótesis, no un resultado. El",
            "contraejemplo conocido es CsPbI₃: pasa el filtro geométrico y su",
            "fase estable a 25 °C es la δ, sin comportamiento de perovskita.",
            "",
            f"De los {min(30, len(status['frontier']))} mostrados, {n_riesgo} están",
            "marcados `marginal` por factor de tolerancia lejos de 1.",
            "",
            "Confirmar la fase exige fonones — frecuencias reales y positivas",
            "indican un mínimo verdadero; una imaginaria señala que el material",
            "quiere caer a otra estructura:",
            "",
            "```bash",
            "buho calc step phonons --phase <material>",
            "```",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"report": str(report_path), "ledger": str(self.ledger_path), "frontier": str(self.frontier_path)}
