"""Prepara directorios y scripts para las relajaciones DFT del cribado.

El funcional es PBE. El fichero de salida se sigue llamando `r2scan.txt`
por compatibilidad: el recolector lo busca por ese nombre y renombrarlo
dejaría ilegibles las corridas que ya están en disco.

Crea para cada candidato:
    runs/relax_basic/{candidate_id}/
    ├── structure.cif
    ├── POSCAR
    ├── input.py          # PBE via GPAWCalculatorFactory + FIRE
    ├── metadata.json
    ├── run.sh            # script de ejecución
    ├── status.json       # {"status": "pending", ...}
    └── generator_config.yaml

Principios:
  - No sobrescribe si status.json existe con status != "pending"
  - Copia la config usada para reproducibilidad
  - Template de input.py importa GPAWCalculatorFactory del proyecto
  - Para superceldas (mixtas): reduce k-points a [2,2,2]

Uso:
    prep = RelaxationJobPreparer(config, project_root=ROOT)
    prep.prepare(scored_candidates, out_root=Path("runs/relax_basic"))
"""
from __future__ import annotations

import json
import logging
import shutil
import string
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from buho.generator.heuristic_generator import GeneratedCandidate
from buho.scoring.pre_dft_score import ScoredCandidate
from buho.structure.build_abx3 import ABX3StructureBuilder

log = logging.getLogger(__name__)

# Usa string.Template ($var) para evitar conflictos con llaves de Python en el código generado
_INPUT_TEMPLATE = string.Template('''\
"""Relajación DFT básica — PBE — generada por BUHO.

Material  : $formula  |  Sn=$has_sn  |  MA=$has_ma  |  supercell=$is_supercell  |  cores=$n_cores
Candidate : $candidate_id
Generado  : $generated_at

CONFIG DE CRIBADO PBE SINGLE-POINT (validado 2026-06-05, GPAW 24.6.0 conda):
  - XC=PBE single-point (max_steps=0): la etiqueta alimenta un GENERADOR, no un
    optimizador → importa el ranking relativo, no el mínimo geométrico exacto.
  - GPAW 24.6.0 (conda env gpaw246), NO master: master crasheaba domain>1.
    En 24.6 domain decomposition escala ~lineal (5.53x en 8 cores).
  - parallel: domain=$n_cores para superceldas (Γ-only). Óptimo de barrido:
    5 slots x 8 cores (domain=8) → throughput 0.749 iters/s, ETA 482 ≈ 5h.
  - ecut=$ecut eV (cribado). k-points: Γ-only superceldas / [2,2,2] puras.
  - convergencia: density=1e-3, eigenstates=1e-4, energy=1e-4. Mixer beta=0.10.
  - Sn: width=0.2 eV (suaviza estados cerca de E_F); sin Hubbard U en cribado.
  - Datasets PAW: GPAW_SETUP_PATH apunta a los del venv (vía runner).
"""
from pathlib import Path
from datetime import datetime, timezone
from ase.io import read
from ase.optimize import FIRE
from gpaw import GPAW, PW, FermiDirac
from gpaw.mixer import Mixer

# Γ-only para superceldas (40 átomos): la celda 2x2x2 pliega la malla [2,2,2]
# primitiva en Γ → 1 k-pt suficiente para relajación básica.
_kpts = [1, 1, 1] if $is_supercell else [2, 2, 2]

# GPAW 24.6 (estable, conda): domain decomposition FUNCIONA en modo PW. GPAW
# master (25.7.1b1) crasheaba con `assert c==1` → estábamos limitados a 1 core.
# En 24.6 domain escala ~lineal (5.53x en 8 cores). Barrido 2026-06-05:
# óptimo de throughput = 5 slots x 8 cores (domain=8). Superceldas Γ-only →
# domain=$n_cores. Puras (4 k-pts irreducibles) → kpt hasta 4, resto a domain.
if $is_supercell:
    _parallel = {"domain": $n_cores, "kpt": 1, "band": 1}
else:
    _kpt = min($n_cores, 4)
    _parallel = {"kpt": _kpt, "domain": max(1, $n_cores // _kpt), "band": 1}

# Ocupaciones: Sn con width amplio (0.2 eV) para suavizar oscilación SCF.
_occ_width = 0.2 if $has_sn else $smearing
# Mixer Pulay beta=0.10 (test 2026-06-05: 28 iters vs 33 con 0.05, misma E).
_mixer = Mixer(0.10, 8, 50)

calc = GPAW(
    mode=PW($ecut),
    xc="PBE",
    kpts={"size": _kpts, "gamma": True},
    occupations=FermiDirac(_occ_width),
    convergence={"density": 1e-3, "eigenstates": 1e-4, "energy": 1e-4},
    mixer=_mixer,
    parallel=_parallel,
    maxiter=$maxiter,
    txt="r2scan.txt",
)

atoms = read("structure.cif")
atoms.calc = calc

from ase.io import write as ase_write
import json, time

# CRIBADO POR SINGLE-POINT (max_steps<=0): la etiqueta DFT alimenta un
# GENERADOR/surrogate, no un optimizador. Lo que importa es el ranking
# relativo de estabilidad con metodología consistente, no el mínimo
# geométrico exacto. SCF single-point = energía electrónica exacta para la
# geometría idealizada (cúbica, lattice_est) — ~5× más barato que FIRE.
# Si max_steps>0 se relaja con FIRE (modo refinamiento).
_max_steps = $max_steps

t0 = time.time()
fmax_final = None
if _max_steps and _max_steps > 0:
    opt = FIRE(atoms, trajectory="relax.traj", logfile="relax.log")
    try:
        converged = opt.run(fmax=$fmax, steps=_max_steps)
        import numpy as _np
        fmax_final = float(_np.linalg.norm(atoms.get_forces(), axis=1).max())
    except Exception as exc:
        Path("error.txt").write_text(str(exc))
        converged = False
    e_total = float(atoms.get_potential_energy())
    mode_label = "relax_fire"
else:
    # Single-point: solo SCF, sin mover átomos.
    try:
        e_total = float(atoms.get_potential_energy())
        converged = bool(getattr(atoms.calc, "scf", None) and atoms.calc.scf.converged) \
            if hasattr(atoms.calc, "scf") else True
    except Exception as exc:
        Path("error.txt").write_text(str(exc))
        e_total = float("nan")
        converged = False
    mode_label = "single_point"

ase_write("relaxed.cif", atoms)

try:
    atoms.calc.write("relaxed.gpw")
except Exception:
    pass

elapsed = time.time() - t0
import math as _math
_e_valid = isinstance(e_total, float) and not _math.isnan(e_total)
status = {
    "status": "converged" if (converged is not False and _e_valid) else "failed",
    "final_energy_eV": e_total if _e_valid else None,
    "energy_per_atom_eV": (e_total / len(atoms)) if _e_valid else None,
    "n_atoms": len(atoms),
    "forces_max_eVA": fmax_final,
    "xc_method": "PBE",
    "ecut_eV": $ecut,
    "kpts": _kpts,
    "fidelity": "relax_basic",
    "calc_mode": mode_label,
    "elapsed_s": round(elapsed, 1),
    "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
Path("status.json").write_text(json.dumps(status, indent=2))
print(f"Done. E={e_total} eV  t={elapsed:.0f}s  mode={mode_label}  converged={converged}")
''')

_RUN_TEMPLATE = '''\
#!/bin/bash
# Relajación PBE del cribado: {formula}
# Generado por BUHO

cd "$(dirname "$0")"
mpirun -n {n_cores} {python} input.py
'''


def _raiz_o_fallo(project_root, quien: str):
    r"""Raíz explícita, o el CWD solo fuera del binario.

    Congelado, el directorio de trabajo no significa nada: al abrir la app desde
    un acceso directo de Windows es `C:\Windows\System32`. Ahí no hay
    proyecto ni permisos de escritura, y el fallo aparece mucho después y
    disfrazado. Si el llamador no pasa raíz, es un error de programación.
    """
    if project_root:
        return Path(project_root)
    if getattr(sys, "frozen", False):
        raise ValueError(
            f"{quien} necesita project_root explícito en el binario: "
            "el directorio de trabajo no es la raíz del proyecto."
        )
    return Path.cwd()


class RelaxationJobPreparer:
    """Prepara directorios de trabajo para relajaciones DFT básicas.

    Parámetros
    ----------
    config      : dict completo de generator.yaml
    project_root: raíz del proyecto (para resolver rutas de src/ y config/)
    n_cores     : número de cores MPI para run.sh (default: 1)
    python      : ejecutable Python (default: python3)
    """

    def __init__(
        self,
        config: dict,
        project_root: Optional[Path] = None,
        n_cores: int = 1,
        python: str = "python3",
        selector_fase=None,
    ):
        self._cfg = config
        self._root = _raiz_o_fallo(project_root, "prepare_relaxation_jobs")
        self._n_cores = n_cores
        self._python = python
        self._dft = config.get("dft_basic", {})
        self._builder = ABX3StructureBuilder(config, random_seed=config.get("random_seed", 42))
        # Selector de fase. Sin el, se prepara la cubica de siempre --- el
        # comportamiento anterior--- porque elegir fase necesita un potencial
        # interatomico y no todas las instalaciones lo tienen. Con el, la
        # estructura que va a DFT es la fase de menor energia, reescalada al
        # tamano de la semilla (ver buho.structure.fases.seleccionar_fase).
        self._selector_fase = selector_fase

        # Detectar si existe el config de GPAW del proyecto
        self._gpaw_config = self._root / "configs" / "default_params.yaml"

    def prepare(
        self,
        candidates: list,  # list[GeneratedCandidate] o list[ScoredCandidate]
        out_root: Optional[Path] = None,
        config_src: Optional[Path] = None,
    ) -> list[Path]:
        """Crea directorios de job para cada candidato.

        Returns
        -------
        Lista de rutas creadas (solo las nuevas, no las saltadas).
        """
        if out_root is None:
            out_root = Path(self._cfg.get("paths", {}).get("relax_dir", "runs/relax_basic"))

        prepared = []
        skipped = 0

        for item in candidates:
            c: GeneratedCandidate = item.candidate if isinstance(item, ScoredCandidate) else item
            job_dir = out_root / c.candidate_id

            if self._should_skip(job_dir):
                skipped += 1
                continue

            atoms, meta = self._builder.build(c, out_dir=job_dir, export=True)
            atoms, meta = self._elegir_fase(c, atoms, meta, job_dir)
            self._write_input(job_dir, c, atoms)
            self._write_run_sh(job_dir, c)
            self._write_status_pending(job_dir, c)

            if config_src is None:
                config_src = self._root / "config" / "generator.yaml"
            if config_src.exists():
                shutil.copy2(config_src, job_dir / "generator_config.yaml")

            prepared.append(job_dir)

        print(f"Preparados: {len(prepared)}  Saltados: {skipped}")
        return prepared

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _elegir_fase(self, candidato, atoms, meta: dict, job_dir: Path):
        """Sustituye la cubica por la fase de menor energia, si hay selector.

        El pipeline construia siempre Pm-3m. Para CsPbI3 eso es la fase alfa,
        que solo existe por encima de 330 C: a temperatura ambiente el material
        esta en otra, con otro bandgap. Medido con el potencial, la cubica queda
        124 meV/f.u. por encima de la ortorrombica.

        Un fallo aqui no puede tumbar la ronda: se avisa y se sigue con la
        cubica, que es lo que habia antes.
        """
        if self._selector_fase is None:
            return atoms, meta
        try:
            r = self._selector_fase(candidato, atoms, meta)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: la seleccion de fase fallo (%s: %s); se usa la cubica",
                        candidato.candidate_id, type(exc).__name__, exc)
            return atoms, meta
        if not r or not r.get("ok") or r.get("atoms") is None:
            log.warning("%s: sin fase elegida (%s); se usa la cubica",
                        candidato.candidate_id, (r or {}).get("motivo", "sin motivo"))
            return atoms, meta

        grupo = r.get("grupo_espacial") or {}
        meta = dict(meta)
        meta.update({
            "fase": r.get("fase"),
            "glazer": r.get("glazer"),
            "grupo_espacial": grupo.get("simbolo"),
            "grupo_espacial_numero": grupo.get("numero"),
            # El tamano viene de la semilla y la forma del potencial: dejar los
            # dos a la vista es lo que permite discutir de donde sale la celda.
            "a_semilla_A": r.get("a_semilla_A"),
            "a_relajado_mlff_A": r.get("a_relajado_mlff_A"),
            "fase_convergida": r.get("convergido"),
            "fases_evaluadas": r.get("ranking"),
        })
        try:
            (job_dir / "fases.json").write_text(
                json.dumps({k: v for k, v in r.items() if k != "atoms"},
                           indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
        except OSError as exc:
            log.warning("%s: no se pudo guardar fases.json: %s",
                        candidato.candidate_id, exc)
        return r["atoms"], meta

    def _should_skip(self, job_dir: Path) -> bool:
        status_file = job_dir / "status.json"
        if not status_file.exists():
            return False
        try:
            st = json.loads(status_file.read_text())
            return st.get("status", "pending") != "pending"
        except Exception:
            return False

    def _write_input(self, job_dir: Path, c: GeneratedCandidate, atoms) -> None:
        is_mixed = (
            len(c.A_site_species) > 1
            or len(c.B_site_species) > 1
            or len(c.X_site_species) > 1
        )
        is_supercell = is_mixed and self._cfg.get("structure", {}).get("supercell_mixed", [1,1,1]) != [1,1,1]
        has_sn = any(sp == "Sn" for sp in c.B_site_species)
        has_ma = any(sp == "MA" for sp in c.A_site_species)

        dft = self._dft
        kpts = [int(k) for k in dft.get("kpts", [4, 4, 4])]
        sn_u_ev = float(dft.get("sn_u_ev", 2.5))
        kpt_rank_cap = int(dft.get("kpt_rank_cap", 4))

        script = _INPUT_TEMPLATE.substitute(
            formula=c.formula,
            candidate_id=c.candidate_id,
            generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            src_path=str(self._root / "src"),
            config_path=str(self._gpaw_config) if self._gpaw_config.exists() else "",
            is_supercell=str(is_supercell),
            has_sn=str(has_sn),
            has_ma=str(has_ma),
            sn_u_ev=sn_u_ev,
            kpt_rank_cap=kpt_rank_cap,
            n_cores=int(self._n_cores),
            ecut=int(dft.get("ecut", 400)),
            kpts=kpts,
            smearing=float(dft.get("smearing", 0.01)),
            maxiter=int(dft.get("maxiter", 300)),
            conv_density=float(dft.get("conv_density", 1e-4)),
            fmax=float(dft.get("fmax", 0.05)),
            max_steps=int(dft.get("max_steps", 200)),
        )
        (job_dir / "input.py").write_text(script, encoding="utf-8")

    def _write_run_sh(self, job_dir: Path, c: GeneratedCandidate) -> None:
        script = _RUN_TEMPLATE.format(
            formula=c.formula,
            n_cores=self._n_cores,
            python=self._python,
        )
        run_sh = job_dir / "run.sh"
        run_sh.write_text(script, encoding="utf-8")
        run_sh.chmod(0o755)

    @staticmethod
    def _write_status_pending(job_dir: Path, c: GeneratedCandidate) -> None:
        status = {
            "status": "pending",
            "candidate_id": c.candidate_id,
            "formula": c.formula,
            "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        (job_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
