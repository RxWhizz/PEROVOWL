"""Cascada de cribado barato para batches de candidatos ABX3.

Cada candidato pasa tiers de costo creciente ANTES del DFT caro:

  Tier 0 — descriptores físicos (instantáneo): PhysicalFilter
           (especies, neutralidad, Goldschmidt t, octaédrico μ, volumen).
  Tier 1 — surrogate (ms): SurrogateEnsemble de bandgap → Eg_pred ± σ,
           band_score (proximidad a la ventana PV), incertidumbre para UCB.
  Tier 2 — MLFF (seg): construir estructura → MEGNet/M3GNet energía de
           formación (Eform) → estabilidad. Indicador de "físicamente
           correcto" sin DFT. Descarta Eform > umbral.

Score de adquisición (active learning):
  total = band_score(PV) + stab_score(Eform) + β·σ(UCB exploración)

Reusa: filters/physical_filters.PhysicalFilter, structure/build_abx3,
ml_surrogate.model.SurrogateEnsemble, ml_surrogate.gnn_predictor.GNNPredictor.
NO requiere DFT.

Tier 2 no tiene por qué correr en este intérprete: torch/matgl/pymatgen pesan
~2 GB y en Windows son la parte más frágil de la pila, así que `buho.mlff_runtime`
decide si se evalúa aquí mismo o en un proceso aparte (WSL). Los dos caminos
devuelven exactamente la misma forma de resultado; el resto de la cascada no
sabe cuál se usó.
"""
from __future__ import annotations

import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from buho import eg_scale
from buho.filters.physical_filters import PhysicalFilter
from buho.generator.heuristic_generator import GeneratedCandidate
from buho.mlff_runtime import MLFFUnavailableError, resolve as resolve_mlff
from buho.structure.build_abx3 import ABX3StructureBuilder
from ml_surrogate.features import (
    CHARGES,
    ELECTRONEG,
    IONIC_RADII,
    ORGANIC_A,
    build_X,
    goldschmidt,
    lattice_est,
    octahedral_factor,
)

log = logging.getLogger(__name__)

_PV_CENTER = 1.45
_PV_SIGMA = 0.35
_STAB_SCALE = 0.5


_PV_MIN_DEFAULT = 1.1
_PV_MAX_DEFAULT = 1.8


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


def _band_score(eg: float) -> float:
    """Gaussiana centrada en el óptimo fotovoltaico (1.45 eV)."""
    if eg is None or math.isnan(eg):
        return 0.0
    return float(np.exp(-0.5 * ((eg - _PV_CENTER) / _PV_SIGMA) ** 2))


def _stab_score(eform: Optional[float]) -> float:
    """sigmoid(−Eform/escala): Eform negativo (estable) → score alto."""
    if eform is None or (isinstance(eform, float) and math.isnan(eform)):
        return 0.5
    return float(1.0 / (1.0 + math.exp(eform / _STAB_SCALE)))


class ScreeningCascade:
    """Orquesta Tier 0-2 sobre un batch de GeneratedCandidate.

    Parámetros
    ----------
    config       : dict completo de generator.yaml
    project_root : raíz del proyecto (para resolver model_path)
    mlff_runtime : dónde evaluar Tier 2. Por defecto se resuelve de la config;
                   los tests lo inyectan para no depender de la máquina.
    """

    def __init__(self, config: dict, project_root: Optional[Path] = None,
                 mlff_runtime=None, extra_roots: tuple[Path, ...] = ()):
        self._cfg = config
        self._root = _raiz_o_fallo(project_root, "ScreeningCascade")
        # Raíces adicionales SOLO para leer modelos. Los de fábrica viajan
        # dentro del binario y los reentrenados se escriben en la raíz de datos:
        # sin mirar en los dos sitios, uno de los dos se pierde.
        self._extra_roots = tuple(Path(r) for r in extra_roots if r)
        self._mlff_runtime = mlff_runtime
        self._scr = config.get("screening", {})
        self._acq = config.get("acquisition", {})
        self._filter = PhysicalFilter(config)
        self._builder = ABX3StructureBuilder(config, random_seed=config.get("random_seed", 42))

        self._beta = float(self._acq.get("beta", 1.0))
        self._eform_max = float(self._scr.get("eform_max_eV_atom", 0.20))
        self._use_surrogate = bool(self._scr.get("tier1_surrogate", True))
        self._use_mlff = bool(self._scr.get("tier2_mlff", True))

        # ── Torre de cribado ─────────────────────────────────────────────────
        # Cada tier estrecha de verdad: el siguiente solo evalúa lo que
        # sobrevivió. Ahí está el ahorro — el Tier 2 cuesta ~0.5 s por candidato
        # frente a los ~2 ms del Tier 1.
        self._tier1_gate = bool(self._scr.get("tier1_gate", True))
        self._tier2_gate = bool(self._scr.get("tier2_gate", True))

        # Holgura de la malla, en desviaciones estándar del propio modelo. El
        # surrogate tiene MAE ≈ 0.31 eV y la ventana PV mide 0.7 eV de ancho:
        # cribar por la estimación puntual tiraría materiales cuyo Eg real sí
        # cae dentro. Con sigma_k=0 la malla es dura; el default deja pasar lo
        # que el modelo no sabe descartar con confianza.
        self._sigma_k = float(self._scr.get("sigma_k", 1.0))

        pv = self._acq.get("pv_window", [_PV_MIN_DEFAULT, _PV_MAX_DEFAULT])
        self._pv_min, self._pv_max = float(pv[0]), float(pv[1])

        self._surrogate = None  # lazy
        self._surrogate_path = None
        self._gnn = None        # lazy

    # ── Carga perezosa de modelos ─────────────────────────────────────────────
    #: Modelo que deja cada reentrenamiento del bucle de descubrimiento. Se
    #: prefiere al de fábrica: si no, el bucle reentrena cada ronda y sigue
    #: cribando con el modelo original — aprende y tira lo aprendido.
    SURROGATE_ACTUAL = ("models", "discovery", "surrogate_bandgap_current.pkl")
    SURROGATE_BASE = ("models", "surrogate_bandgap.pkl")

    def _load_surrogate(self):
        if self._surrogate is not None or not self._use_surrogate:
            return self._surrogate
        from ml_surrogate.model import SurrogateEnsemble

        # Por cada modelo, se mira primero la raíz principal y luego las extra.
        # El reentrenado gana al de fábrica esté donde esté.
        raices = (self._root, *self._extra_roots)
        candidatos = [
            raiz.joinpath(*rel)
            for rel in (self.SURROGATE_ACTUAL, self.SURROGATE_BASE)
            for raiz in raices
        ]
        for mp in candidatos:
            if not mp.exists():
                continue
            try:
                self._surrogate = SurrogateEnsemble.load(mp)
                self._surrogate_path = mp
                return self._surrogate
            except Exception as e:
                # Un modelo corrupto no debe dejar sin Tier 1: se avisa y se
                # prueba el siguiente. Por log ademas de `warnings`: en un
                # servicio o en la GUI un warning de Python no va a ninguna
                # parte, y quedarse sin Tier 1 cambia que candidatos pasan.
                log.warning("surrogate no cargo desde %s (%s: %s)", mp, type(e).__name__, e)
                warnings.warn(f"Cascade: surrogate no cargó desde {mp} ({e}).")
        log.warning(
            "sin surrogate en %s; Tier 1 omitido y el cribado pierde el filtro "
            "de bandgap", " ni ".join(str(c) for c in candidatos),
        )
        warnings.warn(
            "Cascade: no se encontró ningún surrogate en "
            f"{' ni '.join(str(c) for c in candidatos)}; Tier 1 omitido."
        )
        return self._surrogate

    def _runtime(self):
        if self._mlff_runtime is None:
            self._mlff_runtime = resolve_mlff(self._cfg, project_root=self._root)
        return self._mlff_runtime

    def _load_gnn(self):
        """Predictor GNN en este mismo proceso (backend `local`)."""
        if self._gnn is not None:
            return self._gnn
        try:
            from ml_surrogate.gnn_predictor import GNNPredictor
        except ImportError as exc:
            # Traducido a un error tipado: la cascada y el engine tienen que
            # poder degradar a Tier 0/1 sin confundir "falta el entorno MLFF"
            # con un bug de importación cualquiera.
            raise MLFFUnavailableError(
                f"Tier 2 no puede importarse en este intérprete: {exc}",
                remediation="Ejecuta 'buho setup install mlff' o usa el backend WSL.",
            ) from exc
        self._gnn = GNNPredictor(device="cpu")
        return self._gnn

    # ── Tier 2: los dos caminos, misma forma de salida ────────────────────────

    def _tier2_local(self, targets: list[tuple[int, GeneratedCandidate]]) -> dict[int, dict]:
        gnn = self._load_gnn()
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:
            raise MLFFUnavailableError(
                f"Tier 2 necesita pymatgen y no está instalado: {exc}",
                remediation="Ejecuta 'buho setup install mlff'.",
            ) from exc

        out: dict[int, dict] = {}
        for i, candidate in targets:
            try:
                atoms, meta = self._builder.build(candidate, out_dir=None, export=False)
                origen = "pseudoatom" if meta.get("molecular_A_placeholder") else "cubic"
                structure = AseAtomsAdaptor.get_structure(atoms)
            except Exception as exc:
                out[i] = {"error": f"no se pudo construir la estructura: {exc}"}
                continue
            try:
                res = gnn.predict(structure, structure_source=origen)
            except Exception as exc:
                out[i] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            out[i] = {
                "Eg_gnn_eV": res.Eg_eV,
                "Eform_megnet_eV_atom": res.Eform_megnet_eV_atom,
                "Eform_m3gnet_eV_atom": res.Eform_m3gnet_eV_atom,
                "structure_source": res.structure_source,
                "error": None,
            }
        return out

    def _tier2_remote(self, targets: list[tuple[int, GeneratedCandidate]]) -> dict[int, dict]:
        """Delega el lote entero a un proceso con el entorno MLFF (p. ej. WSL)."""
        runtime = self._runtime()
        by_id: dict[str, int] = {}
        payload: list[dict] = []
        for i, candidate in targets:
            cid = str(candidate.candidate_id)
            # Un id repetido haría que el segundo resultado pisara al primero.
            # Es imposible por construcción (el id es hash de la composición),
            # pero si pasara preferimos perder la fila a mezclar dos materiales.
            if cid in by_id:
                continue
            by_id[cid] = i
            payload.append(candidate.to_dict())

        respuesta = runtime.predict(payload, self._cfg)

        out: dict[int, dict] = {}
        for item in respuesta:
            idx = by_id.get(str(item.get("candidate_id", "")))
            if idx is None:
                continue
            out[idx] = item
        for i, _ in targets:
            out.setdefault(i, {"error": "el worker MLFF no devolvió resultado"})
        return out

    def _tier2_predictions(self, targets: list[tuple[int, GeneratedCandidate]]) -> dict[int, dict]:
        if not targets:
            return {}
        runtime = self._runtime()
        if runtime.backend == "off":
            raise MLFFUnavailableError(
                "El tier MLFF está desactivado por configuración.",
                remediation="Pon discovery.mlff.backend en 'auto' en config/generator.yaml.",
            )
        if runtime.backend == "local":
            return self._tier2_local(targets)
        return self._tier2_remote(targets)

    # ── Features desde un candidato mixto (radios efectivos) ──────────────────
    @staticmethod
    def _features(c: GeneratedCandidate,
                  eform: Optional[float] = None,
                  eg_gga: Optional[float] = None) -> dict:
        def eff(fr: dict, table: dict) -> float:
            return sum(f * table[s] for s, f in fr.items())
        A, B, X = c.fractions["A"], c.fractions["B"], c.fractions["X"]
        r_A, r_B, r_X = eff(A, IONIC_RADII), eff(B, IONIC_RADII), eff(X, IONIC_RADII)
        chi_A, chi_B, chi_X = eff(A, ELECTRONEG), eff(B, ELECTRONEG), eff(X, ELECTRONEG)
        q_A, q_B, q_X = eff(A, CHARGES), eff(B, CHARGES), eff(X, CHARGES)
        t = goldschmidt(r_A, r_B, r_X)
        mu = octahedral_factor(r_B, r_X)
        a = lattice_est(r_B, r_X)
        nan = float("nan")
        return {
            "r_A": r_A, "r_B": r_B, "r_X": r_X,
            "chi_A": chi_A, "chi_B": chi_B, "chi_X": chi_X,
            "q_A": q_A, "q_B": q_B, "q_X": q_X,
            "tolerance_t": t, "oct_factor": mu,
            "a_lat_est_A": a, "vol_est_A3": a ** 3,
            "delta_chi_BX": chi_X - chi_B, "mu_BX": r_B + r_X,
            "is_organic_A": float(c.is_organic_A),
            "a_lat_mp_A": nan,
            "band_gap_gga_eV": eg_gga if eg_gga is not None else nan,
            "Eform_eV_atom": eform if eform is not None else nan,
        }

    # ── Pipeline ──────────────────────────────────────────────────────────────
    def screen(self, candidates: list[GeneratedCandidate],
               run_mlff: Optional[bool] = None) -> pd.DataFrame:
        """Corre la cascada y devuelve un DataFrame rankeado con indicadores.

        Columnas: candidate_id, formula, generation_mode, is_organic_A,
        tolerance_t, oct_factor, vol_est_A3, Eg_surrogate_eV, Eg_sigma_eV,
        band_score, in_pv_window, Eg_gnn_eV, Eform_eV_atom, Eform_std_eV_atom,
        is_stable, stab_score, ucb_bonus, total_score, passed_eform, tier_reached,
        dropped_at_tier, drop_reason.

        Es una torre de cribado: cada tier descarta y el siguiente solo evalúa
        lo que sobrevivió. Las filas descartadas SIGUEN en el DataFrame, con
        `dropped_at_tier` y `drop_reason` — se necesita la traza para auditar
        por qué se fue cada material, y `batch_loop` la vuelca a
        cascade_scores.csv. Quien quiera solo los supervivientes usa
        `select_for_dft()`.
        """
        use_mlff = self._use_mlff if run_mlff is None else run_mlff

        # ── Tier 0: descriptores físicos ─────────────────────────────────────
        fr = self._filter.apply(candidates)
        passed = fr.passed
        rows: list[dict] = []
        for c in passed:
            rows.append({
                "candidate_id": c.candidate_id,
                "formula": c.formula,
                "generation_mode": c.generation_mode,
                "is_organic_A": c.is_organic_A,
                "tolerance_t": c.tolerance_t,
                "riesgo_politipo": self._filter.riesgo_politipo(c.tolerance_t),
                "oct_factor": c.oct_factor,
                "vol_est_A3": c.vol_est_A3,
                "Eg_surrogate_eV": None, "Eg_exp_eV": None,
                "eg_scale_delta_eV": None, "Eg_sigma_eV": None,
                "band_score": 0.0, "in_pv_window": None,
                "Eg_gnn_eV": None, "Eform_eV_atom": None,
                "Eform_std_eV_atom": None, "is_stable": None,
                "stab_score": 0.5, "ucb_bonus": 0.0, "total_score": 0.0,
                "passed_eform": True, "tier_reached": 0,
                "dropped_at_tier": None, "drop_reason": None,
            })

        if not passed:
            return pd.DataFrame(rows)

        feats = [self._features(c) for c in passed]

        # ── Tier 1: surrogate bandgap (composición) ──────────────────────────
        sur = self._load_surrogate()
        if sur is not None:
            try:
                Xm = build_X(pd.DataFrame(feats), sur.feature_cols)
                means, stds = sur.predict_batch(Xm)
            except Exception as e:
                warnings.warn(f"Cascade Tier1 falló: {e}")
                means = [float("nan")] * len(passed)
                stds = [float("nan")] * len(passed)
            # La correccion solo vale para un modelo entrenado en la escala que
            # la calibracion asume. Aplicarla a uno viejo no es corregir a
            # medias: mueve las predicciones a un rango arbitrario y el cribado
            # sigue descartandolo todo, ahora por el extremo contrario y con
            # aspecto de estar bien.
            corregir_escala = eg_scale.aplicable_a(sur)
            if not corregir_escala:
                log.warning(
                    "el surrogate no declara la escala '%s': se criba con el "
                    "valor calculado sin llevarlo a escala experimental, y la "
                    "ventana PV no es comparable. Reentrena para corregirlo.",
                    eg_scale.ESCALA,
                )

            for row, cand, mu, sd in zip(rows, passed, means, stds):
                eg_calc = float(mu)
                sigma = float(sd) if not math.isnan(float(sd)) else 0.0
                # El surrogate predice en la escala del calculo (PBE+SOC); la
                # ventana PV esta definida sobre el gap medido. Se guardan los
                # dos valores y el desplazamiento: convertir en silencio dejaria
                # sin forma de auditar de donde sale el numero que decide.
                eg = (eg_scale.a_experimental(
                          eg_calc, cand.fractions["B"], cand.fractions["X"])
                      if corregir_escala else eg_calc)
                row["Eg_surrogate_eV"] = round(eg_calc, 4)
                row["Eg_exp_eV"] = round(eg, 4)
                row["eg_scale_delta_eV"] = round(eg - eg_calc, 4)
                row["Eg_sigma_eV"] = round(float(sd), 4)
                row["band_score"] = round(_band_score(eg), 4)
                row["in_pv_window"] = bool(self._pv_min <= eg <= self._pv_max)
                row["ucb_bonus"] = round(self._beta * sigma, 4)
                row["tier_reached"] = 1

                if not self._tier1_gate:
                    continue
                # Se descarta solo si la ventana no es alcanzable ni contando el
                # margen de error del modelo. Un Eg de 0.95 ± 0.18 sigue siendo
                # un candidato plausible a 1.1 eV.
                margen = self._sigma_k * sigma
                if math.isnan(eg) or eg + margen < self._pv_min or eg - margen > self._pv_max:
                    row["dropped_at_tier"] = 1
                    row["drop_reason"] = (
                        f"Eg {eg:.2f}±{sigma:.2f} eV fuera de la ventana "
                        f"[{self._pv_min}, {self._pv_max}] con {self._sigma_k}σ de holgura"
                        if not math.isnan(eg) else "el surrogate no predijo Eg"
                    )

        # ── Tier 2: MLFF energía de formación / estabilidad ──────────────────
        if use_mlff:
            # Solo los que siguen vivos: construir estructura y evaluar el
            # MLFF cuesta ~0.5 s por candidato, y es exactamente lo que la
            # torre existe para no gastar en material ya descartado.
            targets = [
                (i, passed[i]) for i, row in enumerate(rows)
                if row["dropped_at_tier"] is None
            ]
            # MLFFUnavailableError NO se captura aquí: que falte el entorno MLFF es
            # una condición del sistema, no de un candidato, y quien llama
            # (el engine) tiene que poder distinguirla para degradar a Tier 0/1
            # en vez de marcar 5000 materiales como fallidos.
            predicciones = self._tier2_predictions(targets)

            for i, _ in targets:
                row = rows[i]
                pred = predicciones.get(i) or {}
                if pred.get("error"):
                    motivo = str(pred["error"])
                    if motivo.startswith("no se pudo construir la estructura"):
                        row["dropped_at_tier"] = 2
                        row["drop_reason"] = motivo
                    else:
                        # Antes era un `pass` mudo: un fallo del MLFF dejaba la
                        # fila sin estabilidad y sin decir por qué.
                        warnings.warn(f"Cascade Tier2 falló en {row['formula']}: {motivo}")
                    continue

                eg = pred.get("Eg_gnn_eV")
                if eg is not None and not math.isnan(float(eg)):
                    row["Eg_gnn_eV"] = round(float(eg), 4)
                # Combinar Eform robustamente: MEGNet-Eform a veces da NaN
                # (el ensemble de GNNResult lo propaga). Promediar solo
                # valores finitos de MEGNet/M3GNet.
                vals = [
                    float(v) for v in (pred.get("Eform_megnet_eV_atom"),
                                       pred.get("Eform_m3gnet_eV_atom"))
                    if v is not None and not math.isnan(float(v))
                ]
                if vals:
                    eform = sum(vals) / len(vals)
                    row["Eform_eV_atom"] = round(eform, 4)
                    row["Eform_std_eV_atom"] = (round(abs(vals[0] - vals[1]) / 2, 4)
                                                if len(vals) == 2 else None)
                    row["is_stable"] = bool(eform < -0.1)
                    row["stab_score"] = round(_stab_score(eform), 4)
                    row["passed_eform"] = bool(eform <= self._eform_max)

                    if self._tier2_gate and not row["passed_eform"]:
                        # Misma holgura que arriba: no se tira un material por
                        # un Eform que el propio ensemble no sabe fijar mejor
                        # que su discrepancia.
                        std = row["Eform_std_eV_atom"] or 0.0
                        if eform - self._sigma_k * std > self._eform_max:
                            row["dropped_at_tier"] = 2
                            row["drop_reason"] = (
                                f"E_form {eform:.3f}±{std:.3f} eV/át por encima "
                                f"del umbral {self._eform_max}"
                            )
                        else:
                            row["passed_eform"] = True
                row["tier_reached"] = 2

        # ── Score de adquisición ─────────────────────────────────────────────
        for row in rows:
            row["total_score"] = round(
                row["band_score"] + row["stab_score"] + row["ucb_bonus"], 4)

        # Los descartados por física entran al final con su motivo: sin ellos
        # cascade_scores.csv no permitiría auditar por qué se fue un material.
        for c, motivo in fr.rejected:
            rows.append({
                "candidate_id": c.candidate_id,
                "formula": c.formula,
                "generation_mode": c.generation_mode,
                "is_organic_A": c.is_organic_A,
                "tolerance_t": c.tolerance_t,
                "riesgo_politipo": self._filter.riesgo_politipo(c.tolerance_t),
                "oct_factor": c.oct_factor,
                "vol_est_A3": c.vol_est_A3,
                "Eg_surrogate_eV": None, "Eg_exp_eV": None,
                "eg_scale_delta_eV": None, "Eg_sigma_eV": None,
                "band_score": 0.0, "in_pv_window": None,
                "Eg_gnn_eV": None, "Eform_eV_atom": None,
                "Eform_std_eV_atom": None, "is_stable": None,
                "stab_score": 0.5, "ucb_bonus": 0.0, "total_score": 0.0,
                "passed_eform": False, "tier_reached": 0,
                "dropped_at_tier": 0, "drop_reason": motivo,
            })

        df = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
        return df

    def select_for_dft(self, df: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
        """Selecciona candidatos para DFT.

        Filtra por estabilidad (passed_eform). Si `n` (o config
        `screening.n_dft_per_batch`) es <= 0 o None → devuelve TODOS los que pasen
        estabilidad (rankeados por score). Si n > 0 → solo el top-N.
        """
        if n is None:
            n = int(self._scr.get("n_dft_per_batch", 0))

        keep = df
        if "dropped_at_tier" in keep:
            keep = keep[keep["dropped_at_tier"].isna()]
        keep = keep[keep["passed_eform"]].copy().reset_index(drop=True)
        if n and n > 0:
            keep = keep.head(n)
        return keep.reset_index(drop=True)
