"""Generador heurístico de candidatos ABX3.

Genera composiciones puras y mixtas en los sitios A, B y X de perovskitas
haluro. No inventa etiquetas — solo propone composiciones y descriptores
estructurales derivados de radios iónicos y electronegatividades publicadas.

Uso:
    from buho.generator.heuristic_generator import HeuristicGenerator
    gen = HeuristicGenerator("config/generator.yaml")
    candidates = gen.generate()
    gen.save_jsonl(candidates, "data/raw/generated_candidates.jsonl")
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from buho.structure.occupancy import ajustar_fraccion, sitios_por_subred
from ml_surrogate.features import (
    CHARGES,
    ELECTRONEG,
    IONIC_RADII,
    goldschmidt,
    lattice_est,
    octahedral_factor,
)

ORGANIC_A = {"MA", "FA"}


@dataclass
class GeneratedCandidate:
    """Un candidato ABX3 generado heurísticamente.

    Campos:
        formula             Fórmula legible, e.g. "Cs0.5FA0.5PbI3"
        reduced_formula     Forma canónica normalizada (ordenada + redondeada)
        composition_dict    Átomos por fórmula unitaria, e.g. {"Cs":0.5,"FA":0.5,"Pb":1,"I":3}
        A_site_species      Lista de cationes A, e.g. ["Cs","FA"]
        B_site_species      Lista de cationes B, e.g. ["Pb"]
        X_site_species      Lista de aniones X, e.g. ["I"]
        fractions           {"A":{sp:f}, "B":{sp:f}, "X":{sp:f}}
        candidate_id        SHA1 de reduced_formula (estable ante reordenamientos)
        generation_mode     "pure"|"A_mixed"|"B_mixed"|"X_mixed"|"multi_mixed"
        tolerance_t         Factor de tolerancia de Goldschmidt
        oct_factor          Factor octaédrico μ = r_B / r_X_eff
        a0_est_A            Parámetro de red estimado (Å)
        vol_est_A3          Volumen estimado por fórmula unitaria (Å³)
        is_organic_A        True si algún A-site es MA o FA
        metadata            Reproducibilidad: fecha, hostname, versión Python, config_hash
    """

    formula: str
    reduced_formula: str
    composition_dict: dict
    A_site_species: list
    B_site_species: list
    X_site_species: list
    fractions: dict
    candidate_id: str
    generation_mode: str
    tolerance_t: float
    oct_factor: float
    a0_est_A: float
    vol_est_A3: float
    is_organic_A: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratedCandidate":
        return cls(**d)


def _canonical_id(fractions: dict, grid: float = 0.001) -> str:
    """SHA1 estable de la composición canónica.

    `grid` define la resolución de dedup: las fracciones se cuantizan a la
    rejilla más cercana antes de hashear. discreto → 0.001 (fracciones exactas);
    continuo → 0.02 (epsilon-cluster: composiciones a <2% se consideran iguales,
    evitando re-DFT de casi-duplicados entre batches).
    """
    parts = []
    for site in ("A", "B", "X"):
        site_fracs = fractions.get(site, {})
        for sp in sorted(site_fracs):
            val = round(round(site_fracs[sp] / grid) * grid, 4)
            parts.append(f"{site}:{sp}:{val}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _effective_r(species: list[str], fracs: dict[str, float], table: dict) -> float:
    return sum(fracs[sp] * table[sp] for sp in species)


def _reduced_formula(fractions: dict) -> str:
    parts = []
    for site in ("A", "B", "X"):
        for sp in sorted(fractions.get(site, {})):
            f = fractions[site][sp]
            parts.append(f"{sp}{f:.3g}" if abs(f - 1.0) > 1e-6 else sp)
    return "".join(parts)


def _build_formula(A_sp: list, B_sp: list, X_sp: list,
                   A_f: dict, B_f: dict, X_f: dict) -> str:
    def site_str(species, fracs):
        if len(species) == 1:
            return species[0]
        return "".join(f"{sp}{fracs[sp]:.2g}" for sp in sorted(species))

    return site_str(A_sp, A_f) + site_str(B_sp, B_f) + site_str(X_sp, X_f) + "3"


def _metadata(config_hash: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "config_hash": config_hash,
    }


class HeuristicGenerator:
    """Genera candidatos ABX3 puros y mixtos desde un archivo de configuración YAML.

    Parámetros
    ----------
    config_path : ruta al YAML de configuración (config/generator.yaml)
    random_seed : semilla para reproducibilidad en modos mixtos
    """

    def __init__(self, config_path: str | Path | dict, random_seed: Optional[int] = None):
        if isinstance(config_path, dict):
            self._cfg = copy.deepcopy(config_path)
            raw = json.dumps(self._cfg, sort_keys=True, default=str).encode()
            self._config_hash = hashlib.sha1(raw).hexdigest()[:8]
        else:
            cfg_path = Path(config_path)
            with open(cfg_path) as f:
                self._cfg = yaml.safe_load(f)
            self._config_hash = hashlib.sha1(cfg_path.read_bytes()).hexdigest()[:8]
        self._seed = random_seed if random_seed is not None else self._cfg.get("random_seed", 42)
        self._rng = np.random.RandomState(self._seed)

        cs = self._cfg["chemical_space"]
        self._A_sites: list[str] = cs["A_sites"]
        self._B_sites: list[str] = cs["B_sites"]
        self._X_sites: list[str] = cs["X_sites"]

        gen = self._cfg["generation"]
        self._fractions: list[float] = [float(f) for f in gen["fractions"]]
        self._modes: dict = gen.get("modes", {
            "pure": True, "A_mixed": True, "B_mixed": True,
            "X_mixed": True, "multi_mixed": False,
        })

        # Muestreo continuo (active learning por batches): en vez de iterar la
        # lista discreta de fracciones, se muestrean `n_samples_per_combo` valores
        # continuos por combinación de sitios. Reproducible vía seed + batch_id.
        self._fraction_mode: str = gen.get("fraction_mode", "discrete")
        self._n_samples: int = int(gen.get("n_samples_per_combo", 8))
        self._batch_size: int = int(gen.get("batch_size", 1000))
        # Resolución de dedup: 0.001 (discreto exacto) / 0.02 (epsilon continuo)
        self._id_grid: float = 0.02 if self._fraction_mode == "continuous" else 0.001

        # Cuántos sitios tiene cada subred en la supercelda que se construirá.
        # El muestreo continuo proponía fracciones como 0.051 que 8 sitios no
        # pueden representar: `round(0.051*8) = 0` átomos, así que se calculaba
        # el endmember y se archivaba con la fórmula del dopado. Ajustando aquí,
        # la fórmula del candidato describe la estructura que se construye.
        # La lista discreta original ([0.125 … 0.875]) ya eran múltiplos de 1/8;
        # esto extiende esa propiedad al modo continuo, y la sigue si se
        # agranda la supercelda.
        st = self._cfg.get("structure", {}) or {}
        self._sitios = sitios_por_subred(st.get("supercell_mixed", [2, 2, 2]))

    # ── Public API ──────────────────────────────────────────────────────────────

    def generate(self) -> list[GeneratedCandidate]:
        """Genera todos los candidatos configurados y elimina duplicados."""
        candidates: list[GeneratedCandidate] = []

        if self._modes.get("pure", True):
            candidates.extend(self._generate_pure())

        if self._modes.get("A_mixed", True):
            candidates.extend(self._generate_A_mixed())

        if self._modes.get("B_mixed", True):
            candidates.extend(self._generate_B_mixed())

        if self._modes.get("X_mixed", True):
            candidates.extend(self._generate_X_mixed())

        if self._modes.get("multi_mixed", False):
            candidates.extend(self._generate_multi_mixed())

        return self._deduplicate(candidates)

    def generate_batch(
        self,
        batch_id: int,
        batch_size: Optional[int] = None,
        registry_path: Optional[str | Path] = None,
    ) -> list[GeneratedCandidate]:
        """Genera un batch reproducible de candidatos continuos.

        - RNG sembrado con `seed + batch_id` → mismo batch_id ⇒ mismos candidatos.
        - Dedup dentro del batch y, si se da `registry_path`, contra un registro
          persistente de candidate_ids ya vistos (dedup entre batches). Los IDs
          nuevos se anexan al registro.
        - Trunca a `batch_size` (config `generation.batch_size` por defecto).
        """
        size = batch_size if batch_size is not None else self._batch_size
        self._rng = np.random.RandomState(self._seed + int(batch_id))
        cands = self._deduplicate(self.generate())

        seen = self._load_registry(registry_path) if registry_path else set()
        fresh = [c for c in cands if c.candidate_id not in seen]
        if len(fresh) > size:
            order = self._rng.permutation(len(fresh))
            fresh = [fresh[i] for i in order[:size]]

        if registry_path and fresh:
            self._append_registry(registry_path, [c.candidate_id for c in fresh])
        return fresh

    @staticmethod
    def _load_registry(path: str | Path) -> set[str]:
        p = Path(path)
        if not p.exists():
            return set()
        return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}

    @staticmethod
    def _append_registry(path: str | Path, ids: list[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for cid in ids:
                f.write(cid + "\n")

    # ── Sampling de fracciones (discreto vs continuo) ─────────────────────────────

    def _binary_fracs(self, sitio: str = "A") -> list[float]:
        """Fracciones para una mezcla binaria (la otra es 1-f).

        El ajuste a la rejilla de la supercelda se aplica en los DOS modos.
        Hacerlo solo en el continuo dejaba fuera al protocolo autonomo, que
        entra por el discreto: `ChemicalSpaceEnumerator` sobrescribe
        `generation.fractions` con una rejilla de paso 0.0073, y 30 000 de esas
        composiciones colapsaban en 1 998 estructuras — 15 candidatos por cada
        estructura realmente nueva, y es el camino que gasta el DFT.
        """
        n = self._sitios[sitio]
        if self._fraction_mode == "continuous":
            crudas: list[float] = [
                float(f) for f in self._rng.uniform(0.05, 0.95, self._n_samples)
            ]
        else:
            crudas = [f for f in self._fractions
                      if abs(f) > 1e-9 and abs(f - 1.0) > 1e-9]
        # Deduplicado: varias fracciones crudas caen en el mismo multiplo.
        return sorted({ajustar_fraccion(f, n) for f in crudas})

    def _ternary_x_fracs(self) -> list[tuple[float, float, float]]:
        """Fracciones (xI, xBr, xCl) para mezcla ternaria de haluros.

        Como en las binarias, se ajusta en ambos modos: la tercera fracción se
        deriva de las otras dos para que sumen exactamente 1, y se descartan las
        combinaciones donde a esa tercera no le quedaría ni un átomo.
        """
        n = self._sitios["X"]
        if self._fraction_mode == "continuous":
            crudas = [(float(a), float(b))
                      for a, b, _ in self._rng.dirichlet([1.0, 1.0, 1.0], self._n_samples)]
        else:
            crudas = [(xI, xBr) for xI in self._fractions for xBr in self._fractions]

        out = set()
        for a, b in crudas:
            a_aj = ajustar_fraccion(a, n)
            b_aj = ajustar_fraccion(b, n)
            c_aj = 1.0 - a_aj - b_aj
            if c_aj < 1.0 / n - 1e-9:   # la tercera especie no cabría
                continue
            out.add((a_aj, b_aj, c_aj))
        return sorted(out)

    def _multi_fracs(self) -> list[tuple[float, float]]:
        """Pares (fA, fX) para multi_mixed (A y X simultáneos)."""
        if self._fraction_mode == "continuous":
            crudos = list(zip(self._rng.uniform(0.05, 0.95, self._n_samples),
                              self._rng.uniform(0.05, 0.95, self._n_samples)))
        else:
            base = [f for f in self._fractions
                    if abs(f) > 1e-9 and abs(f - 1.0) > 1e-9]
            crudos = list(itertools.product(base, base))
        return sorted({
            (ajustar_fraccion(float(a), self._sitios["A"]),
             ajustar_fraccion(float(x), self._sitios["X"]))
            for a, x in crudos
        })

    @staticmethod
    def save_jsonl(candidates: list[GeneratedCandidate], path: str | Path) -> None:
        """Guarda candidatos en formato JSONL (un JSON por línea)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for c in candidates:
                f.write(json.dumps(c.to_dict()) + "\n")

    @staticmethod
    def load_jsonl(path: str | Path) -> list[GeneratedCandidate]:
        """Carga candidatos desde JSONL."""
        candidates = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(GeneratedCandidate.from_dict(json.loads(line)))
        return candidates

    # ── Generation modes ────────────────────────────────────────────────────────

    def _generate_pure(self) -> list[GeneratedCandidate]:
        result = []
        for A, B, X in itertools.product(self._A_sites, self._B_sites, self._X_sites):
            c = self._make_candidate(
                A_sp=[A], B_sp=[B], X_sp=[X],
                A_f={A: 1.0}, B_f={B: 1.0}, X_f={X: 1.0},
                mode="pure",
            )
            if c is not None:
                result.append(c)
        return result

    def _generate_A_mixed(self) -> list[GeneratedCandidate]:
        result = []
        for (A1, A2), B, X in itertools.product(
            itertools.combinations(self._A_sites, 2),
            self._B_sites,
            self._X_sites,
        ):
            for f in self._binary_fracs("A"):
                c = self._make_candidate(
                    A_sp=[A1, A2], B_sp=[B], X_sp=[X],
                    A_f={A1: f, A2: 1.0 - f},
                    B_f={B: 1.0}, X_f={X: 1.0},
                    mode="A_mixed",
                )
                if c is not None:
                    result.append(c)
        return result

    def _generate_B_mixed(self) -> list[GeneratedCandidate]:
        result = []
        for A, (B1, B2), X in itertools.product(
            self._A_sites,
            itertools.combinations(self._B_sites, 2),
            self._X_sites,
        ):
            for f in self._binary_fracs("B"):
                c = self._make_candidate(
                    A_sp=[A], B_sp=[B1, B2], X_sp=[X],
                    A_f={A: 1.0},
                    B_f={B1: f, B2: 1.0 - f},
                    X_f={X: 1.0},
                    mode="B_mixed",
                )
                if c is not None:
                    result.append(c)
        return result

    def _generate_X_mixed(self) -> list[GeneratedCandidate]:
        result = []
        # 2-component X mixtures
        for A, B, (X1, X2) in itertools.product(
            self._A_sites,
            self._B_sites,
            itertools.combinations(self._X_sites, 2),
        ):
            for f in self._binary_fracs("X"):
                c = self._make_candidate(
                    A_sp=[A], B_sp=[B], X_sp=[X1, X2],
                    A_f={A: 1.0}, B_f={B: 1.0},
                    X_f={X1: f, X2: 1.0 - f},
                    mode="X_mixed",
                )
                if c is not None:
                    result.append(c)
        # 3-component X mixtures (I+Br+Cl): grid discreta o simplex Dirichlet
        if len(self._X_sites) >= 3:
            for A, B in itertools.product(self._A_sites, self._B_sites):
                for xI, xBr, xCl in self._ternary_x_fracs():
                    if xCl < 0.05 or xCl > 0.95:
                        continue
                    X_sp = [x for x, f in zip(
                        self._X_sites, [xI, xBr, xCl]) if f > 0.01]
                    # Sin redondear: las fracciones ya vienen ajustadas a los
                    # 24 sitios X, y round(17/24, 4) + round(2/24, 4) +
                    # round(5/24, 4) = 0.9999, que rompe la suma a 1.
                    X_f = {x: f for x, f in zip(
                        self._X_sites, [xI, xBr, xCl]) if f > 0.01}
                    if len(X_sp) < 3:
                        continue  # already covered by 2-component
                    c = self._make_candidate(
                        A_sp=[A], B_sp=[B], X_sp=X_sp,
                        A_f={A: 1.0}, B_f={B: 1.0}, X_f=X_f,
                        mode="X_mixed",
                    )
                    if c is not None:
                        result.append(c)
        return result

    def _generate_multi_mixed(self) -> list[GeneratedCandidate]:
        """A-site + X-site mixtos simultáneamente."""
        result = []
        for (A1, A2), B, (X1, X2) in itertools.product(
            itertools.combinations(self._A_sites, 2),
            self._B_sites,
            itertools.combinations(self._X_sites, 2),
        ):
            for fA, fX in self._multi_fracs():
                c = self._make_candidate(
                    A_sp=[A1, A2], B_sp=[B], X_sp=[X1, X2],
                    A_f={A1: fA, A2: 1.0 - fA},
                    B_f={B: 1.0},
                    X_f={X1: fX, X2: 1.0 - fX},
                    mode="multi_mixed",
                )
                if c is not None:
                    result.append(c)
        return result

    # ── Core builder ────────────────────────────────────────────────────────────

    def _make_candidate(
        self,
        A_sp: list[str], B_sp: list[str], X_sp: list[str],
        A_f: dict[str, float], B_f: dict[str, float], X_f: dict[str, float],
        mode: str,
    ) -> Optional[GeneratedCandidate]:
        """Construye un candidato verificando carga y disponibilidad de datos."""
        # Verificar que todas las especies tienen datos
        for sp in A_sp + B_sp + X_sp:
            if sp not in IONIC_RADII or sp not in CHARGES:
                return None

        # Neutralidad de carga: sum(q_A * f_A) + sum(q_B * f_B) + 3*sum(q_X * f_X) = 0
        q_A_eff = sum(CHARGES[sp] * A_f[sp] for sp in A_sp)
        q_B_eff = sum(CHARGES[sp] * B_f[sp] for sp in B_sp)
        q_X_eff = sum(CHARGES[sp] * X_f[sp] for sp in X_sp)
        if abs(q_A_eff + q_B_eff + 3.0 * q_X_eff) > 0.05:
            return None

        # Radios efectivos
        r_A = _effective_r(A_sp, A_f, IONIC_RADII)
        r_B = _effective_r(B_sp, B_f, IONIC_RADII)
        r_X = _effective_r(X_sp, X_f, IONIC_RADII)

        t = goldschmidt(r_A, r_B, r_X)
        mu = octahedral_factor(r_B, r_X)
        a0 = lattice_est(r_B, r_X)
        vol = a0 ** 3

        fractions = {"A": dict(A_f), "B": dict(B_f), "X": dict(X_f)}
        cid = _canonical_id(fractions, self._id_grid)
        red = _reduced_formula(fractions)
        formula = _build_formula(A_sp, B_sp, X_sp, A_f, B_f, X_f)

        comp = {}
        for sp, f in A_f.items():
            comp[sp] = comp.get(sp, 0.0) + f
        for sp, f in B_f.items():
            comp[sp] = comp.get(sp, 0.0) + f
        for sp, f in X_f.items():
            comp[sp] = comp.get(sp, 0.0) + 3.0 * f

        return GeneratedCandidate(
            formula=formula,
            reduced_formula=red,
            composition_dict=comp,
            A_site_species=list(A_sp),
            B_site_species=list(B_sp),
            X_site_species=list(X_sp),
            fractions=fractions,
            candidate_id=cid,
            generation_mode=mode,
            tolerance_t=round(t, 5),
            oct_factor=round(mu, 5),
            a0_est_A=round(a0, 4),
            vol_est_A3=round(vol, 3),
            is_organic_A=any(sp in ORGANIC_A for sp in A_sp),
            metadata=_metadata(self._config_hash),
        )

    @staticmethod
    def _deduplicate(candidates: list[GeneratedCandidate]) -> list[GeneratedCandidate]:
        seen: set[str] = set()
        unique = []
        for c in candidates:
            if c.candidate_id not in seen:
                seen.add(c.candidate_id)
                unique.append(c)
        return unique
