"""Corrección scissor del bandgap de cribado, por elemento del sitio B.

El problema
-----------
El cribado etiqueta candidatos con un bandgap de PBE **sin acoplamiento
espín-órbita**, y luego lo compara contra una ventana fotovoltaica derivada del
límite de Shockley-Queisser, que se calcula sobre el bandgap real. Son
magnitudes distintas, y la diferencia no es un desplazamiento constante: el SOC
es un efecto relativista que crece con el número atómico del catión B, cuyos
orbitales p forman el mínimo de la banda de conducción.

Medido sobre CsBI₃ cúbico con los mismos parámetros del cribado
(`scripts/calibrate_soc_scissor.py`):

    χ_SOC(Pb) = −0.630 eV      (Z=82)
    χ_SOC(Ge) = −0.221 eV      (Z=32)
    χ_SOC(Sn) = −0.061 eV      (Z=50)

Aplicar un valor único a toda la familia sesgaría la comparación entre
elementos B en más de medio electrón-voltio.

Lo que esto corrige y lo que no
-------------------------------
Corrige la omisión del SOC, que es real, medida y dependiente del elemento.

**No** corrige otras dos fuentes de error del mismo número, ambas medidas y
ninguna resuelta aquí:

1. *Fase*. El cribado calcula el gap de una perovskita cúbica ideal. Para
   CsSnI₃ (ortorrómbica con octaedros inclinados a temperatura ambiente) el
   cúbico da Eg(PBE) = 0.26 eV frente a 1.3 eV experimentales. La inclinación
   octaédrica abre el gap, y aquí nunca ocurre. Es el hallazgo 7.2 de
   `docs/metodologia-dft.md` apareciendo como error de bandgap.
2. *Intercambio-correlación*. PBE subestima gaps por error de
   autointeracción; corregirlo exige un híbrido (HSE06), no calibrado aquí.

Por eso el resultado se guarda en una columna aparte y la cruda se conserva:
un número corregido a medias no debe pasar por medido.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Tabla por defecto, relativa a la raíz que la contenga.
TABLA_REL = "config/soc_scissor.json"

_cache: dict[str, dict[str, float]] = {}


def _raices() -> list[Path]:
    """Raíces donde puede vivir la tabla, en orden de preferencia.

    Congelado con PyInstaller, `__file__` cuelga del directorio de extracción
    (`sys._MEIPASS`), así que `parents[2]` apunta un nivel POR ENCIMA del bundle
    y no encuentra nada. Era el caso: en los binarios publicados la tabla no se
    cargaba y `corregir()` devolvía el gap de PBE sin tocar — en silencio.
    """
    raices: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        raices.append(Path(meipass))
    # Desde el código fuente: src/buho/bandgap_scissor.py -> raíz del repo.
    raices.append(Path(__file__).resolve().parents[2])
    return raices


def cargar_tabla(ruta: Path | str | None = None) -> dict[str, float]:
    """Lee `chi_soc_eV` de la tabla de calibración. Vacío si no existe."""
    if ruta is not None:
        candidatas = [Path(ruta)]
    else:
        candidatas = [raiz / TABLA_REL for raiz in _raices()]

    clave = str(candidatas[0])
    if clave in _cache:
        return _cache[clave]

    tabla: dict[str, Any] = {}
    encontrada = next((c for c in candidatas if c.is_file()), None)
    if encontrada is not None:
        try:
            datos = json.loads(encontrada.read_text(encoding="utf-8"))
            tabla = {k: float(v) for k, v in (datos.get("chi_soc_eV") or {}).items()}
            # Por (B, X) cuando esta: el SOC depende del haluro tanto como del
            # metal --- medido, CsPbI3 da -0.630 y CsPbBr3 -1.340 con el mismo Pb.
            bx = datos.get("chi_soc_bx_eV") or {}
            if bx:
                tabla["_bx"] = {b: {x: float(v) for x, v in por_x.items()}
                                for b, por_x in bx.items()}
        except (OSError, ValueError, TypeError) as exc:
            log.warning("tabla de scissor ilegible en %s: %s", encontrada, exc)
    else:
        # A nivel WARNING, no INFO: que la corrección no se aplique cambia las
        # etiquetas de entrenamiento, y antes se perdía entre el ruido.
        log.warning(
            "sin tabla de scissor en %s; el bandgap NO se corrige por SOC",
            " ni ".join(str(c) for c in candidatas),
        )

    _cache[clave] = tabla
    return tabla


def chi_soc(fracciones_b: dict[str, float],
            tabla: dict[str, Any] | None = None,
            fracciones_x: dict[str, float] | None = None) -> float:
    """χ_SOC del candidato, ponderado por ocupación de los dos sitios.

    Una composición mezclada interpola entre los χ de sus elementos, igual que
    se interpolan los radios. Un elemento sin calibrar aporta 0: es preferible
    corregir de menos que inventar el valor.

    Con `fracciones_x` y una tabla que traiga `chi_soc_bx_eV`, el χ sale de la
    pareja (B, X). Sin ellas cae a la tabla por-B, que es el yoduro de
    referencia. La diferencia no es un matiz: para Pb, el bromuro tiene el
    doble de corrección que el yoduro.
    """
    tabla = cargar_tabla() if tabla is None else tabla
    if not tabla or not fracciones_b:
        return 0.0
    bx = tabla.get("_bx") or {}
    if bx and fracciones_x:
        total = 0.0
        for b, fb in fracciones_b.items():
            por_x = bx.get(b)
            if por_x is None:
                total += float(fb) * float(tabla.get(b, 0.0) or 0.0)
                continue
            for x, fx in fracciones_x.items():
                total += float(fb) * float(fx) * float(por_x.get(x, 0.0))
        return total
    return sum(f * float(tabla.get(sp, 0.0) or 0.0)
               for sp, f in fracciones_b.items() if sp != "_bx")


def corregir(eg_pbe: float | None, fracciones_b: dict[str, float],
             tabla: dict[str, Any] | None = None,
             fracciones_x: dict[str, float] | None = None) -> float | None:
    """Bandgap de cribado con el SOC sumado. `None` entra y sale como `None`.

    No se recorta a cero a propósito: un gap corregido negativo significa que
    el material sale metálico a este nivel de teoría, y esconderlo tras un
    max(0, ·) haría pasar por semiconductor lo que el cálculo dice que no lo es.
    """
    if eg_pbe is None:
        return None
    return float(eg_pbe) + chi_soc(fracciones_b, tabla, fracciones_x)


def describir(tabla: dict[str, float] | None = None) -> dict[str, Any]:
    """Resumen para diagnóstico e informes."""
    tabla = cargar_tabla() if tabla is None else tabla
    return {
        "disponible": bool(tabla),
        "chi_soc_eV": dict(tabla),
        "corrige": ["acoplamiento espín-órbita, por elemento del sitio B"],
        "no_corrige": [
            "fase: el cribado usa la perovskita cúbica ideal, no la fase real",
            "intercambio-correlación: PBE subestima el gap (haría falta HSE06)",
        ],
    }
