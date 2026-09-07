"""Fases candidatas de una perovskita ABX3, en vez de suponer que es cubica.

Por que existe
--------------
El pipeline construia siempre Pm-3m y fijaba el parametro de red con una formula
empirica. El propio codigo ya avisaba de que eso no basta: `riesgo_politipo`
señala que el factor de Goldschmidt es condicion necesaria y no suficiente, y
pone el contraejemplo de casa --- CsPbI3 tiene t = 0.851, dentro de rango, y su
fase estable a 25 C es la delta ortorrombica, con octaedros de arista compartida
y Eg ~ 2.82 eV. La alfa cubica que el pipeline evaluaba solo existe por encima de
330 C.

Este modulo genera las fases que esta familia admite de verdad, para que compitan
en energia y gane la que corresponda, en lugar de decidirlo por decreto.

Que genera y que no
-------------------
Las estructuras que salen de aqui son **puntos de partida para relajar**, no
resultados. Los desplazamientos de las inclinaciones se aplican por octaedro y
pueden dejar pequenas inconsistencias en los aniones compartidos entre dos
octaedros; la relajacion posterior las limpia. Lo que importa es sembrar cuencas
de simetria distintas, no que la semilla sea exacta.

Quien decide en que fase quedo el material es `grupo_espacial`, sobre la
estructura YA relajada. Nunca la etiqueta con la que se genero.

Sistemas de inclinacion (notacion de Glazer)
--------------------------------------------
El anion X esta compartido entre dos octaedros BX6 vecinos, asi que inclinar uno
obliga al de al lado. Solo hay dos formas de propagarse a lo largo de un eje:

  * en fase (+): octaedros consecutivos giran en el MISMO sentido.
  * en antifase (-): giran en sentidos opuestos.

De ahi salen las fases que se observan en haluros: I4/mcm (a0a0c-),
P4/mbm (a0a0c+), R-3c (a-a-a-) y Pnma (a-a-c+), ademas de la cubica sin
inclinar.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Amplitud inicial de la inclinacion, en grados. No es el valor final --- eso
#: lo decide la relajacion--- sino un empujon suficiente para salir de la cuenca
#: cubica sin deformar tanto que la relajacion no sepa volver.
ANGULO_SEMILLA = 8.0


@dataclass(frozen=True)
class Fase:
    """Una fase candidata: como sembrarla y como se llama."""

    nombre: str
    #: Notacion de Glazer, o None para las que no son distorsiones de la cubica.
    glazer: str | None
    #: Grupo espacial que se ESPERA obtener. Es documentacion, no una promesa:
    #: la relajacion puede acabar en otro sitio, y eso lo dice `grupo_espacial`.
    grupo_esperado: str
    #: Amplitud por eje (x, y, z) en grados, y si cada eje va en fase o antifase.
    tilts: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fases_eje: tuple[str, str, str] = ("0", "0", "0")   # "0" | "+" | "-"
    #: Supercelda del padre cubico necesaria para que la inclinacion quepa.
    supercelda: tuple[int, int, int] = (1, 1, 1)


#: Las fases de inclinacion que se observan en perovskitas de haluro. No es la
#: lista completa de los 23 sistemas de Glazer: son las que aparecen en esta
#: familia, y anadir mas cuesta una relajacion cada una.
FASES_INCLINACION: tuple[Fase, ...] = (
    Fase("cubica", "a0a0a0", "Pm-3m", (0.0, 0.0, 0.0), ("0", "0", "0"), (1, 1, 1)),
    Fase("tetragonal_antifase", "a0a0c-", "I4/mcm",
         (0.0, 0.0, ANGULO_SEMILLA), ("0", "0", "-"), (2, 2, 2)),
    Fase("tetragonal_en_fase", "a0a0c+", "P4/mbm",
         (0.0, 0.0, ANGULO_SEMILLA), ("0", "0", "+"), (2, 2, 2)),
    Fase("romboedrica", "a-a-a-", "R-3c",
         (ANGULO_SEMILLA,) * 3, ("-", "-", "-"), (2, 2, 2)),
    Fase("ortorrombica", "a-a-c+", "Pnma",
         (ANGULO_SEMILLA, ANGULO_SEMILLA, ANGULO_SEMILLA), ("-", "-", "+"), (2, 2, 2)),
)


def _signo_giro(indice_celda: tuple[int, int, int], eje: int, modo: str) -> float:
    """Sentido del giro del octaedro de la celda (i, j, k) alrededor de `eje`.

    La regla sale de como se propaga la inclinacion por la red: en antifase el
    sentido alterna en las tres direcciones a la vez, y en fase alterna solo en
    las dos perpendiculares al eje de giro --- a lo largo del propio eje, los
    octaedros giran igual, que es lo que "en fase" significa.
    """
    i, j, k = indice_celda
    if modo == "0":
        return 0.0
    if modo == "-":
        return -1.0 if (i + j + k) % 2 else 1.0
    # En fase: la paridad se cuenta solo en los dos ejes perpendiculares.
    perpendiculares = [(j + k), (i + k), (i + j)][eje]
    return -1.0 if perpendiculares % 2 else 1.0


def _matriz_giro(omega: np.ndarray) -> np.ndarray:
    """Rotacion de Rodrigues alrededor de `omega`, cuyo modulo es el angulo.

    Se compone UNA sola rotacion en vez de encadenar tres sobre x, y y z: las
    rotaciones no conmutan, y encadenarlas no da la inclinacion que se pretende.
    Con a-a-a-, por ejemplo, el giro correcto es uno solo alrededor de [111];
    encadenar tres bajaba la simetria a P-1.
    """
    angulo = float(np.linalg.norm(omega))
    if angulo < 1e-12:
        return np.eye(3)
    k = omega / angulo
    kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    return (np.eye(3) + math.sin(angulo) * kx
            + (1.0 - math.cos(angulo)) * (kx @ kx))


def aplicar_inclinacion(atoms, fase: Fase, *, b_sites: set[str],
                        x_sites: set[str]):
    """Devuelve una copia de `atoms` con la inclinacion de `fase` sembrada.

    Se itera sobre los OCTAEDROS, no sobre los aniones. Cada anion puente tiene
    dos cationes B a la misma distancia, asi que preguntarle "cual es el mas
    cercano" da una respuesta numericamente arbitraria y rompe el patron: era el
    fallo que dejaba las fases inclinadas en P1.

    Iterando por octaedro no hay ambiguedad, y la escritura doble no molesta:
    dos octaedros vecinos contrarrotan por construccion --- es lo que impone
    compartir vertice--- asi que los dos calculan la MISMA posicion para el
    anion que comparten.
    """
    if all(m == "0" for m in fase.fases_eje):
        return atoms.copy()

    trabajo = atoms.copy()
    celda = np.array(trabajo.cell)
    inversa = np.linalg.inv(celda.T)
    simbolos = trabajo.get_chemical_symbols()
    posiciones = trabajo.get_positions()

    indices_b = [n for n, s in enumerate(simbolos) if s in b_sites]
    indices_x = [n for n, s in enumerate(simbolos) if s in x_sites]
    if not indices_b or not indices_x:
        log.warning("fase %s: no hay sitios B o X reconocibles; se deja sin inclinar",
                    fase.nombre)
        return trabajo

    n_sc = np.array(fase.supercelda, dtype=float)
    escalados = trabajo.get_scaled_positions()
    pos_x = posiciones[indices_x]

    # Lado del octaedro: la mitad de la arista de la celda cubica.
    lado = float(np.linalg.norm(celda[0]) / n_sc[0])
    radio = lado * 0.5 * 1.25   # margen para no perder vecinos por redondeo

    # Se ACUMULA el desplazamiento que cada octaedro pide para sus aniones y
    # luego se promedia. Con giros de un solo eje los dos octaedros que comparten
    # un anion piden lo mismo y el promedio no cambia nada; con giros de varios
    # ejes ya no contrarrotan alrededor de un eje unico, piden cosas distintas, y
    # sobrescribir dejaba el anion donde dijera el ultimo que pasara por ahi ---
    # una eleccion arbitraria que hundia la simetria hasta P-1. Promediar es la
    # forma sensata de repartir la restriccion de compartir vertice.
    acumulado = np.zeros_like(posiciones)
    cuenta = np.zeros(len(posiciones))
    for idx_b in indices_b:
        centro = posiciones[idx_b]
        celda_idx = tuple(
            int(math.floor(f * n + 1e-6)) % int(n)
            for f, n in zip(escalados[idx_b], n_sc)
        )
        # Vector de giro del octaedro: la suma de las contribuciones de cada eje,
        # aplicada como una sola rotacion.
        omega = np.zeros(3)
        for eje in range(3):
            modo = fase.fases_eje[eje]
            grados = fase.tilts[eje]
            if modo == "0" or grados == 0.0:
                continue
            omega[eje] = math.radians(grados) * _signo_giro(celda_idx, eje, modo)
        if not np.any(omega):
            continue
        giro = _matriz_giro(omega)

        # Los seis aniones del octaedro, con imagen minima.
        d = pos_x - centro
        frac = d @ inversa.T
        frac -= np.round(frac)
        d = frac @ celda
        dist = np.linalg.norm(d, axis=1)
        for local, n_global in enumerate(indices_x):
            if dist[local] > radio:
                continue
            # El DESPLAZAMIENTO, no la posicion: dos octaedros a lados
            # opuestos de la celda ven el mismo anion como imagenes periodicas
            # distintas, y promediar posiciones absolutas lo mandaba al centro
            # de la celda. El desplazamiento por giro es el mismo en cualquier
            # imagen.
            acumulado[n_global] += giro @ d[local] - d[local]
            cuenta[n_global] += 1.0

    nuevas = posiciones.copy()
    movidos = cuenta > 0
    nuevas[movidos] += acumulado[movidos] / cuenta[movidos, None]
    trabajo.set_positions(nuevas)
    trabajo.wrap()
    return trabajo


def grupo_espacial(atoms, *, tolerancia: float = 0.05) -> dict[str, Any]:
    """Grupo espacial de una estructura, medido --- no supuesto.

    `tolerancia` es generosa a proposito: sobre una estructura relajada
    numericamente, exigir simetria exacta devolveria P1 para cosas que son
    claramente tetragonales. Se reporta la tolerancia usada para que el numero
    se pueda interpretar.
    """
    try:
        import spglib
    except ImportError:
        return {"disponible": False, "motivo": "spglib no instalado"}

    celda = (atoms.get_cell()[:], atoms.get_scaled_positions(),
             atoms.get_atomic_numbers())
    try:
        datos = spglib.get_symmetry_dataset(celda, symprec=tolerancia)
    except Exception as exc:  # noqa: BLE001 - spglib puede fallar en celdas raras
        return {"disponible": False, "motivo": f"{type(exc).__name__}: {exc}"}
    if datos is None:
        return {"disponible": False, "motivo": "spglib no identifico simetria"}

    # spglib devuelve dataclass en versiones nuevas y dict en las viejas.
    numero = getattr(datos, "number", None)
    simbolo = getattr(datos, "international", None)
    if numero is None and isinstance(datos, dict):
        numero, simbolo = datos.get("number"), datos.get("international")
    return {
        "disponible": True,
        "numero": int(numero) if numero is not None else None,
        "simbolo": simbolo,
        "tolerancia": tolerancia,
    }


def generar_candidatas(atoms, *, b_sites: set[str], x_sites: set[str],
                       fases: tuple[Fase, ...] = FASES_INCLINACION,
                       supercelda_base: tuple[int, int, int] = (2, 2, 2),
                       ) -> list[dict[str, Any]]:
    """Siembra una estructura por fase candidata.

    `atoms` es el padre cubico ya construido. Se expande a la supercelda que
    cada inclinacion necesita y se aplica el patron de giros.
    """
    from ase.build import make_supercell

    salida: list[dict[str, Any]] = []
    for fase in fases:
        objetivo = tuple(max(a, b) for a, b in zip(fase.supercelda, supercelda_base)) \
            if fase.glazer != "a0a0a0" else fase.supercelda
        base = atoms.copy()
        if objetivo != (1, 1, 1):
            base = make_supercell(base, np.diag(objetivo))
        fase_expandida = Fase(
            fase.nombre, fase.glazer, fase.grupo_esperado,
            fase.tilts, fase.fases_eje, objetivo,
        )
        sembrada = aplicar_inclinacion(
            base, fase_expandida, b_sites=b_sites, x_sites=x_sites)
        salida.append({
            "fase": fase.nombre,
            "glazer": fase.glazer,
            "grupo_esperado": fase.grupo_esperado,
            "supercelda": list(objetivo),
            "n_atomos": len(sembrada),
            "atoms": sembrada,
        })
    return salida


# ── Elegir la fase ───────────────────────────────────────────────────────────

def reescalar_a_volumen(atoms, volumen_objetivo: float):
    """Escala la celda uniformemente hasta un volumen dado.

    El escalado uniforme conserva el grupo espacial exactamente: mueve todos los
    atomos en coordenadas fraccionarias identicas. Sirve para quedarse con la
    FORMA de la distorsion que dio la relajacion y el TAMANO que dice otra
    fuente.
    """
    actual = float(atoms.get_volume())
    if actual <= 0 or volumen_objetivo <= 0:
        return atoms.copy()
    factor = (volumen_objetivo / actual) ** (1.0 / 3.0)
    salida = atoms.copy()
    salida.set_cell(np.array(salida.cell) * factor, scale_atoms=True)
    return salida


def ordenar_por_energia(candidatas: list[dict[str, Any]], calculador, *,
                        fmax: float = 0.05, pasos: int = 300,
                        ) -> list[dict[str, Any]]:
    """Relaja cada candidata y las ordena de menor a mayor energia por formula.

    El potencial decide el ORDEN, que es para lo que sirve. El tamano de celda
    que salga de aqui no se usa: ver `seleccionar_fase`.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    salida = []
    for cand in candidatas:
        est = cand["atoms"].copy()
        n_fu = max(1, len(est) // 5)
        est.calc = calculador
        try:
            convergido = bool(
                FIRE(FrechetCellFilter(est), logfile=None).run(fmax=fmax, steps=pasos))
            energia = float(est.get_potential_energy()) / n_fu
        except Exception as exc:  # noqa: BLE001 - una fase no puede tumbar la ronda
            log.warning("fase %s: la relajacion fallo (%s: %s); se descarta",
                        cand["fase"], type(exc).__name__, exc)
            continue
        finally:
            est.calc = None
        salida.append({**cand, "atoms": est, "n_formulas": n_fu,
                       "convergido": convergido,
                       "E_por_formula_eV": round(energia, 5)})

    salida.sort(key=lambda r: r["E_por_formula_eV"])
    if salida:
        base = salida[0]["E_por_formula_eV"]
        for r in salida:
            r["dE_meV_por_formula"] = round(1000 * (r["E_por_formula_eV"] - base), 1)
    return salida


def seleccionar_fase(atoms, calculador, *, b_sites: set[str], x_sites: set[str],
                     a_semilla: float, supercelda_base: tuple[int, int, int] = (2, 2, 2),
                     ) -> dict[str, Any]:
    """Elige la fase de menor energia y le devuelve el tamano de la semilla.

    Por que se separan las dos cosas
    --------------------------------
    Medido sobre CsPbI3: el potencial ORDENA bien --- pone la cubica 124 meV/f.u.
    por encima de la ortorrombica, que es lo que dice el experimento, porque la
    alfa cubica solo existe por encima de 330 C--- pero el TAMANO de celda que
    devuelve es peor que el de partida: 6.4619 A frente a 6.1834 de la semilla,
    con 6.18 experimental. Un +4.56 % contra un +0.06 %.

    No es que el potencial falle: esta entrenado sobre PBE y reproduce el
    volumen de PBE, que sobreestima el de estos haluros blandos. Pero el gap es
    muy sensible al volumen --- comprimir un 2.1 % costaba 0.35 eV en los
    bromuros--- asi que quedarse con esa celda desharia lo que gana el funcional.

    De ahi el reparto: el potencial aporta la FORMA de la distorsion, que es lo
    que no sabiamos, y la semilla calibrada contra parametros de red
    experimentales aporta el TAMANO, que ya sabiamos. El reescalado es uniforme,
    asi que el grupo espacial no cambia.
    """
    candidatas = generar_candidatas(
        atoms, b_sites=b_sites, x_sites=x_sites, supercelda_base=supercelda_base)
    ordenadas = ordenar_por_energia(candidatas, calculador)
    if not ordenadas:
        return {"ok": False, "motivo": "ninguna fase relajo"}

    ganadora = ordenadas[0]
    n_fu = ganadora["n_formulas"]
    escalada = reescalar_a_volumen(ganadora["atoms"], n_fu * a_semilla ** 3)

    return {
        "ok": True,
        "fase": ganadora["fase"],
        "glazer": ganadora["glazer"],
        "grupo_espacial": grupo_espacial(escalada),
        "convergido": ganadora["convergido"],
        "a_semilla_A": round(a_semilla, 4),
        "a_relajado_mlff_A": round(float(
            (ganadora["atoms"].get_volume() / n_fu) ** (1 / 3)), 4),
        "atoms": escalada,
        "ranking": [
            {k: v for k, v in r.items() if k != "atoms"} for r in ordenadas
        ],
    }
