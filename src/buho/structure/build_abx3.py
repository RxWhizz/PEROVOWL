"""Constructor de estructuras cristalinas ABX3 genéricas.

Reutiliza StructureBuilder de dft_cspbi3 para:
  - Composiciones puras (inorgánicas): cúbica Pm-3m, 5 átomos
  - Composiciones mixtas: supercelda 2×2×2 (40 átomos) con distribución aleatoria

Para A-sites orgánicos (MA, FA):
  - Se sustituye por el placeholder configurado (default: Cs)
  - Se registra en metadata["molecular_A_placeholder"] = True
  - Se añade advertencia explícita en metadata["organic_A_warning"]

Uso:
    builder = ABX3StructureBuilder(config)
    atoms, meta = builder.build(candidate, out_dir=Path("runs/relax_basic/abc123"))
"""
from __future__ import annotations

import math

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from buho.generator.heuristic_generator import GeneratedCandidate
from buho.structure.occupancy import (
    cuantizar_ocupacion,
    fracciones_realizadas,
    sitios_por_subred,
)

log = logging.getLogger(__name__)

ORGANIC_A = {"MA", "FA"}
_ORG_WARNING = (
    "Estructura usa catión orgánico simplificado en posición A. "
    "Reemplazar con geometría molecular explícita antes de DFT de producción."
)

#: Cuánto se acorta el enlace B–X respecto a la suma de radios iónicos, por
#: elemento del sitio B. Los radios iónicos suponen enlace puramente iónico; en
#: estas perovskitas el enlace metal–haluro tiene covalencia apreciable y sale
#: más corto. Calibrado contra estructura experimental donde la hay:
#:
#:   Pb: CsPbI3 a=6.18 Å frente a 6.78 de la fórmula -> 0.912
#:   Sn: CsSnI3 a=6.22 Å frente a 6.76             -> 0.920
#:
#: Ge se deja SIN contraer a propósito. No tengo una referencia experimental
#: verificada para CsGeI3, y aplicarle el factor de Pb/Sn lo sobrecorrige: con
#: 0.915 la celda baja a 5.36 Å y el cálculo sale **metálico** (Eg = 0.00 eV),
#: que es peor que el error original. Con el radio de Ge (0.73 Å, mucho menor
#: que Pb/Sn) la suma de radios ya cae cerca de lo razonable. Un elemento sin
#: entrada aquí no se contrae.
BOND_CONTRACTION: dict[str, float] = {"Pb": 0.912, "Sn": 0.920}


class ABX3StructureBuilder:
    """Construye estructuras ASE para candidatos ABX3 generados.

    Parámetros
    ----------
    config : dict con sección "structure" del YAML del generador
    random_seed : semilla para distribución de especies en superceldas
    """

    def __init__(self, config: dict, random_seed: int = 42):
        st = config.get("structure", {})
        self._supercell_pure: list[int] = list(st.get("supercell_pure", [1, 1, 1]))
        self._supercell_mixed: list[int] = list(st.get("supercell_mixed", [2, 2, 2]))
        self._organic_placeholder: str = st.get("organic_A_placeholder", "Cs")
        self._formats: list[str] = list(st.get("export_formats", ["cif", "poscar", "traj"]))
        # Contracción del enlace B–X por elemento; ver BOND_CONTRACTION. Un
        # dict vacío en la config reproduce el comportamiento anterior.
        contraccion = st.get("bond_contraction")
        self._bond_contraction: dict[str, float] = (
            dict(contraccion) if isinstance(contraccion, dict) else dict(BOND_CONTRACTION)
        )
        self._seed = random_seed

    def build(
        self,
        candidate: GeneratedCandidate,
        out_dir: Optional[Path] = None,
        export: bool = True,
    ) -> tuple:
        """Construye la estructura y opcionalmente la exporta.

        Returns
        -------
        (atoms, metadata)  — ASE Atoms y dict con información del candidato
        """
        from ase.build import make_supercell
        from dft_cspbi3.structure_builder import StructureBuilder
        from ml_surrogate.features import lattice_est, IONIC_RADII

        is_mixed = (
            len(candidate.A_site_species) > 1
            or len(candidate.B_site_species) > 1
            or len(candidate.X_site_species) > 1
        )
        has_organic = candidate.is_organic_A
        self._perdidas_ultima_build: list[str] = []

        # Fracciones que la supercelda puede representar de verdad. La celda se
        # dimensiona con estas y no con las pedidas: dimensionarla para una
        # composición que no contiene deja el tamaño y el contenido en
        # desacuerdo. Cuando el generador propone fracciones representables
        # —que es lo que hace desde el arreglo de `ajustar_fraccion`— ambas
        # coinciden y esto no cambia nada.
        _sitios = sitios_por_subred(self._supercell_mixed if is_mixed else [1, 1, 1])
        fracciones_reales = {
            sitio: fracciones_realizadas(
                list(getattr(candidate, f"{sitio}_site_species", []) or []),
                candidate.fractions.get(sitio, {}) or {},
                _sitios[sitio],
            )
            for sitio in ("A", "B", "X")
        }

        # ── Determine structural A/B/X for crystal building ──────────────────
        # Para A: usar placeholder si hay catión orgánico
        if has_organic:
            A_struct = self._organic_placeholder
        else:
            A_struct = candidate.A_site_species[0]  # caso puro o usaremos primera

        B_struct = candidate.B_site_species[0]

        X_struct = candidate.X_site_species[0]

        # ── Lattice constant ─────────────────────────────────────────────────
        r_B_eff = sum(
            fracciones_reales["B"].get(sp, 0.0) * IONIC_RADII[sp]
            for sp in candidate.B_site_species
        )
        r_X_eff = sum(
            fracciones_reales["X"].get(sp, 0.0) * IONIC_RADII[sp]
            for sp in candidate.X_site_species
        )
        # `lattice_est` devuelve 2√2·(r_B+r_X), que es la relación A–X. La red
        # cúbica Pm-3m se fija por el enlace B–X, que vale a/2, así que
        # a = 2·(r_B+r_X) = lattice_est/√2. Sin dividir, las estructuras salían
        # √2 veces dilatadas —CsSnI3 a 9.56 Å en vez de 6.76— con el armazón
        # B–X a 4.78 Å en lugar de 3.15: sin enlaces que dibujar y, como avisa
        # `phase2_force/prepare.py`, «casi metálicas».
        #
        # Se corrige aquí y no en `lattice_est` porque ese valor es además la
        # característica 11 del surrogate (`a_lat_est_A`), con la que se
        # entrenó el modelo: cambiarla invalidaría las predicciones.
        #
        # La contracción corrige que `r_B + r_X` sobreestima el enlace B–X (ver
        # BOND_CONTRACTION). El exceso no es inocuo: sobre CsPbI3, con la celda
        # dilatada un 9.7 % el Eg de PBE sale 1.78 eV en vez de 1.09 eV. Casi
        # 0.7 eV de error — más que el que introduce ignorar el acoplamiento
        # espín-órbita. Se pondera por fracción igual que los radios, para que
        # una composición con el sitio B mezclado interpole entre sus factores.
        factor = sum(
            fracciones_reales["B"].get(sp, 0.0) * self._bond_contraction.get(sp, 1.0)
            for sp in candidate.B_site_species
        )
        a0 = lattice_est(r_B_eff, r_X_eff) / math.sqrt(2.0) * factor

        # ── Build primitive cell ─────────────────────────────────────────────
        atoms = StructureBuilder.build_perovskite_cubic(A_struct, B_struct, X_struct, a0)

        # ── Supercell ────────────────────────────────────────────────────────
        sc = self._supercell_mixed if is_mixed else self._supercell_pure
        if sc != [1, 1, 1]:
            import numpy as np
            sc_matrix = np.diag(sc)
            atoms = make_supercell(atoms, sc_matrix)

        # ── Distribute mixed species ─────────────────────────────────────────
        if is_mixed:
            atoms = self._distribute_mixed_species(atoms, candidate, sc)

        n_atoms = len(atoms)

        # ── Metadata ─────────────────────────────────────────────────────────
        metadata = {
            "candidate_id": candidate.candidate_id,
            "formula": candidate.formula,
            "reduced_formula": candidate.reduced_formula,
            "generation_mode": candidate.generation_mode,
            "A_site_species": candidate.A_site_species,
            "B_site_species": candidate.B_site_species,
            "X_site_species": candidate.X_site_species,
            "fractions": candidate.fractions,
            # Lo que la supercelda contiene de verdad. Difiere de `fractions`
            # cuando se piden fracciones que 8 (o 24) sitios no pueden
            # representar; sin esto, un resultado de DFT no dice a qué
            # composición corresponde realmente.
            "fractions_realized": fracciones_reales,
            "composition_exact": not self._perdidas_ultima_build
            and all(
                abs(fracciones_reales[s].get(sp, 0.0) - float(candidate.fractions.get(s, {}).get(sp, 0.0))) < 1e-9
                for s in ("A", "B", "X")
                for sp in (getattr(candidate, f"{s}_site_species", []) or [])
            ),
            "species_dropped": list(self._perdidas_ultima_build),
            "molecular_A_placeholder": has_organic,
            "organic_A_warning": _ORG_WARNING if has_organic else None,
            "lattice_constant_A": round(a0, 4),
            "supercell": sc,
            "n_atoms": n_atoms,
            "tolerance_t": candidate.tolerance_t,
            "oct_factor": candidate.oct_factor,
            "random_seed": self._seed,
            "build_date": datetime.utcnow().isoformat() + "Z",
        }

        if out_dir is not None and export:
            self.export(atoms, Path(out_dir), metadata)

        return atoms, metadata

    def export(self, atoms, out_dir: Path, metadata: dict) -> None:
        """Exporta estructura en los formatos configurados."""
        out_dir.mkdir(parents=True, exist_ok=True)

        if "cif" in self._formats:
            atoms.write(str(out_dir / "structure.cif"))

        if "poscar" in self._formats:
            atoms.write(str(out_dir / "POSCAR"), format="vasp")

        if "traj" in self._formats:
            from ase.io.trajectory import Trajectory
            with Trajectory(str(out_dir / "structure.traj"), "w") as traj:
                traj.write(atoms)

        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _distribute_mixed_species(self, atoms, candidate: GeneratedCandidate, sc: list):
        """Distribuye especies mezcladas en los sitios de la supercelda."""
        import numpy as np
        from ase import Atoms

        rng = np.random.RandomState(self._seed)
        symbols = list(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        # Identify site indices by original element (pre-supercell first species)
        # A-site: was A_struct; B-site: was B_struct; X-site: was X_struct
        A_struct = self._organic_placeholder if candidate.is_organic_A else candidate.A_site_species[0]
        B_struct = candidate.B_site_species[0]
        X_struct = candidate.X_site_species[0]

        A_idxs = [i for i, s in enumerate(symbols) if s == A_struct]
        B_idxs = [i for i, s in enumerate(symbols) if s == B_struct]
        X_idxs = [i for i, s in enumerate(symbols) if s == X_struct]

        def _ase_symbol(sp: str) -> str:
            """Convierte especie a símbolo ASE válido (organics → placeholder)."""
            return self._organic_placeholder if sp in ORGANIC_A else sp

        def _assign(idxs, species_list, fracs, sitio):
            if len(species_list) <= 1:
                return
            total = len(idxs)
            pares = cuantizar_ocupacion(list(species_list), fracs, total)
            counts = [n for _, n in pares]

            # Una especie pedida que se queda sin un solo átomo convierte la
            # mezcla en su endmember, pero la fórmula sigue anunciando el
            # dopante. Callarlo hace que el DFT del endmember se archive como si
            # fuera el del dopado.
            perdidas = [sp for sp, n in pares if n == 0 and float(fracs.get(sp, 0.0)) > 0.0]
            if perdidas:
                log.warning(
                    "%s: %s desaparece del sitio %s — %d sitios no pueden "
                    "representar fracciones menores que %.4f; se construye %s",
                    candidate.formula, ", ".join(perdidas), sitio, total,
                    1.0 / (2 * total), "el endmember",
                )
                self._perdidas_ultima_build.extend(perdidas)

            shuffled = idxs.copy()
            rng.shuffle(shuffled)
            offset = 0
            for sp, n in zip(species_list, counts):
                ase_sp = _ase_symbol(sp)
                for idx in shuffled[offset:offset + n]:
                    symbols[idx] = ase_sp
                offset += n

        if len(candidate.A_site_species) > 1:
            _assign(A_idxs, candidate.A_site_species, candidate.fractions["A"], "A")

        if len(candidate.B_site_species) > 1:
            _assign(B_idxs, candidate.B_site_species, candidate.fractions["B"], "B")

        if len(candidate.X_site_species) > 1:
            _assign(X_idxs, candidate.X_site_species, candidate.fractions["X"], "X")

        new_atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True,
        )
        new_atoms.info.update(atoms.info)
        return new_atoms
