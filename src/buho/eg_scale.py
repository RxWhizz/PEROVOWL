"""Lleva el bandgap calculado a la escala experimental.

Por que hace falta
------------------
El cribado predice un bandgap en la escala del calculo (PBE con SOC
perturbativo) y lo compara contra la ventana fotovoltaica [1.1, 1.8] eV, que
sale del limite de Shockley-Queisser y esta definida sobre el gap **medido**.
Son magnitudes distintas y la diferencia no es pequena: medido sobre los
haluros de Cs de referencia, PBE+SOC subestima el gap en mas de 1 eV.

La consecuencia era total, no gradual: de 3793 candidatos del protocolo
autonomo, los 3793 caian en el Tier 1 porque el surrogate predecia alrededor de
0.96 eV y la ventana empieza en 1.1. El bucle cribaba, no encontraba nada
elegible y se declaraba terminado sin lanzar un solo calculo DFT.

Que corrige y que no
--------------------
Esto corrige la **escala**, no el ordenamiento. Un desplazamiento suma lo mismo
a todos los candidatos de una misma pareja (B, X), asi que no puede reordenar
dentro de esa familia: si el nivel de teoria ordena mal dos materiales, esto no
lo arregla. Sirve para que la ventana se aplique sobre magnitudes comparables.

El desplazamiento depende de B **y** de X. Suponerlo constante por elemento B
seria falso: medido, CsPbI3 y CsPbBr3 difieren en 0.83 eV con el mismo B.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TABLA_REL = Path("config") / "eg_scale.json"

#: Nombre de la escala que asume esta calibracion: bandgap de PBE con SOC
#: perturbativo, sobre la geometria con contraccion B-X por pareja.
#:
#: Existe porque un desplazamiento calibrado para una escala aplicado a un
#: modelo entrenado en otra da un numero sin sentido, y no hay forma de notarlo
#: mirando el resultado. Paso de verdad: el modelo publicado predecia ~0.96 eV
#: —etiquetas de PBE sobre la geometria vieja, sin SOC— y sumarle el
#: desplazamiento lo llevaba a 3.04 eV, fuera de la ventana por arriba. Antes
#: estaba fuera por abajo; el cribado seguia descartandolo todo, ahora por el
#: otro extremo y con aspecto de estar corregido.
ESCALA = "pbe_soc_geom_bx"

#: Atributo con el que se sella un modelo entrenado en esta escala.
ATRIBUTO_MODELO = "escala_bandgap"

_cache: dict[str, dict[str, Any]] = {}


def escala_del_modelo(modelo: Any) -> str | None:
    """La escala en la que se entreno un modelo, si la declara."""
    return getattr(modelo, ATRIBUTO_MODELO, None)


def sellar_modelo(modelo: Any) -> Any:
    """Marca el modelo con la escala en la que se acaba de entrenar."""
    setattr(modelo, ATRIBUTO_MODELO, ESCALA)
    return modelo


def aplicable_a(modelo: Any) -> bool:
    """Si tiene sentido corregir la salida de este modelo.

    Un modelo sin sello viene de antes de la calibracion: se entreno con
    etiquetas de otra escala y corregirlo empeoraria el resultado en vez de
    mejorarlo.
    """
    return escala_del_modelo(modelo) == ESCALA


def _raices() -> list[Path]:
    """Raices donde puede vivir la tabla, en orden de preferencia.

    Congelado con PyInstaller, `__file__` cuelga del directorio de extraccion
    (`sys._MEIPASS`), asi que `parents[2]` apunta un nivel POR ENCIMA del bundle.
    Es el fallo que dejo el scissor de SOC sin aplicarse en todos los binarios
    publicados, en silencio; aqui se evita desde el principio.
    """
    raices: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        raices.append(Path(meipass))
    raices.append(Path(__file__).resolve().parents[2])
    return raices


def cargar_tabla(ruta: Path | str | None = None) -> dict[str, Any]:
    """Lee el ajuste de la tabla de calibracion. Vacio si no existe."""
    candidatas = [Path(ruta)] if ruta is not None else [r / TABLA_REL for r in _raices()]
    clave = str(candidatas[0])
    if clave in _cache:
        return _cache[clave]

    ajuste: dict[str, Any] = {}
    encontrada = next((c for c in candidatas if c.is_file()), None)
    if encontrada is not None:
        try:
            datos = json.loads(encontrada.read_text(encoding="utf-8"))
            bruto = datos.get("ajuste") or {}
            if bruto.get("columnas") and bruto.get("coeficientes"):
                ajuste = {
                    "columnas": list(bruto["columnas"]),
                    "coeficientes": [float(v) for v in bruto["coeficientes"]],
                }
                # El error de la propia calibracion viaja con el ajuste. Antes
                # se descartaba al parsear, y la ventana PV se aplicaba como si
                # el desplazamiento fuera exacto: CsPbI3 (1.839 predicho, 1.73
                # medido) y CsSnBr3 (1.844 / 1.75) se descartaban siendo
                # candidatos validos, por menos de lo que la tabla ya sabia que
                # se equivocaba.
                err = (datos.get("errores_pct") or {}).get(
                    "error_relativo_medio_corregido")
                if err is not None:
                    try:
                        ajuste["error_relativo_pct"] = float(err)
                    except (TypeError, ValueError):
                        log.warning("error_relativo_medio_corregido no numerico "
                                    "en %s: %r", encontrada, err)
        except (OSError, ValueError, TypeError) as exc:
            log.warning("tabla de escala ilegible en %s: %s", encontrada, exc)
    else:
        # WARNING y no INFO: sin correccion, la ventana fotovoltaica se aplica
        # sobre una magnitud que no es la suya y descarta candidatos validos.
        log.warning(
            "sin tabla de escala en %s; el bandgap NO se lleva a escala "
            "experimental y la ventana PV se aplicara sobre el valor calculado",
            " ni ".join(str(c) for c in candidatas),
        )

    _cache[clave] = ajuste
    return ajuste


def delta(fracciones_b: dict[str, float], fracciones_x: dict[str, float],
          ajuste: dict[str, Any] | None = None) -> float:
    """Desplazamiento a sumar, ponderado por la ocupacion de ambos sitios.

    Una composicion mixta interpola linealmente entre los desplazamientos de sus
    elementos, igual que se interpolan los radios. Es una aproximacion: la
    calibracion se hizo sobre compuestos puros y no se ha comprobado que
    interpole bien. Un elemento sin calibrar aporta 0 -- corregir de menos es
    preferible a inventar el valor.
    """
    ajuste = cargar_tabla() if ajuste is None else ajuste
    if not ajuste:
        return 0.0
    cols = ajuste["columnas"]
    coef = ajuste["coeficientes"]
    por_nombre = dict(zip(cols, coef))

    total = float(por_nombre.get("c", 0.0))
    for sp, f in (fracciones_b or {}).items():
        total += float(f) * float(por_nombre.get(f"B:{sp}", 0.0))
    for sp, f in (fracciones_x or {}).items():
        total += float(f) * float(por_nombre.get(f"X:{sp}", 0.0))
    return total


def a_experimental(eg_calculado: float | None,
                   fracciones_b: dict[str, float],
                   fracciones_x: dict[str, float],
                   ajuste: dict[str, Any] | None = None) -> float | None:
    """El gap calculado, llevado a la escala en la que esta la ventana PV.

    `None` entra y sale como `None`: que el surrogate no haya predicho nada no
    es lo mismo que predecir cero.
    """
    if eg_calculado is None:
        return None
    return float(eg_calculado) + delta(fracciones_b, fracciones_x, ajuste)


def margen_calibracion(eg: float | None,
                       ajuste: dict[str, Any] | None = None) -> float:
    """Cuanto se equivoca la propia escala en este valor, en eV.

    El error de la tabla es **relativo** (6.0 % medido sobre los nueve haluros
    de referencia), asi que en eV crece con el gap: a 1.8 eV son 0.11 eV, que es
    justo el orden de lo que separaba a los dos falsos negativos de borde de
    entrar en la ventana. Devolver un margen y no ensanchar la ventana es
    deliberado: la ventana [1.1, 1.8] sale de Shockley-Queisser y no depende de
    lo bien que calibremos; lo que depende es la confianza con la que podemos
    decir que un candidato cae fuera.

    Sin tabla el margen es 0.0 --- sin correccion de escala tampoco hay error de
    correccion que contar, y el numero cribado ni siquiera esta en esa escala.
    """
    if eg is None:
        return 0.0
    ajuste = cargar_tabla() if ajuste is None else ajuste
    pct = float(ajuste.get("error_relativo_pct", 0.0) or 0.0)
    if pct <= 0.0:
        return 0.0
    return abs(float(eg)) * pct / 100.0


def describir(ajuste: dict[str, Any] | None = None) -> dict[str, Any]:
    """Estado de la calibracion, para diagnostico."""
    ajuste = cargar_tabla() if ajuste is None else ajuste
    return {
        "disponible": bool(ajuste),
        "columnas": ajuste.get("columnas", []),
        "coeficientes": ajuste.get("coeficientes", []),
        "error_relativo_pct": ajuste.get("error_relativo_pct"),
    }
