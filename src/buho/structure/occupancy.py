"""Ocupación de sitios en la supercelda: de fracciones continuas a átomos enteros.

Una supercelda tiene un número finito de sitios por subred, así que solo puede
representar fracciones que sean múltiplos de `1/n_sitios`. Pedir
`Cs0.95Rb0.051SnI3` en una 2×2×2 (8 sitios A) da `round(0.051 * 8) = 0` átomos de
Rb: la estructura que se calcula es CsSnI3 puro, pero la fórmula sigue diciendo
que lleva Rb. Con eso, el DFT devuelve el bandgap del endmember y se archiva como
si fuera el del dopado.

El efecto no se limita a los dopantes diluidos: toda la composición se cuantiza.
Sobre 40 000 candidatos del pool, las composiciones pedidas colapsaban en 2 204
estructuras distintas — 18 candidatos "nuevos" por cada estructura realmente
nueva.

Este módulo es la única definición de esa cuantización. El constructor la usa
para colocar átomos, el generador para no proponer lo irrepresentable, y el
bucle de descubrimiento para no mandar dos veces la misma estructura a DFT. Si
vivieran en sitios distintos podrían discrepar, que es como empezó el problema.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

# Sitios por celda primitiva ABX3: un A, un B, tres X.
SITIOS_PRIMITIVOS = {"A": 1, "B": 1, "X": 3}


def sitios_por_subred(supercell: Iterable[int]) -> dict[str, int]:
    """Cuántos sitios de cada subred hay en la supercelda."""
    celdas = 1
    for n in supercell:
        celdas *= int(n)
    return {sitio: celdas * base for sitio, base in SITIOS_PRIMITIVOS.items()}


def cuantizar_ocupacion(
    especies: list[str],
    fracciones: Mapping[str, float],
    n_sitios: int,
) -> list[tuple[str, int]]:
    """Reparte `n_sitios` entre `especies` según `fracciones`.

    Devuelve pares (especie, n_atomos) en el orden de `especies`. La última se
    queda con el resto, de modo que la suma es exactamente `n_sitios`.
    """
    if not especies:
        return []
    if len(especies) == 1:
        return [(especies[0], n_sitios)]

    cuentas: list[tuple[str, int]] = []
    quedan = n_sitios
    for especie in especies[:-1]:
        n = min(int(round(float(fracciones.get(especie, 0.0)) * n_sitios)), quedan)
        cuentas.append((especie, n))
        quedan -= n
    cuentas.append((especies[-1], quedan))
    return cuentas


def fracciones_realizadas(
    especies: list[str],
    fracciones: Mapping[str, float],
    n_sitios: int,
) -> dict[str, float]:
    """Las fracciones que la estructura tiene de verdad, no las que se pidieron."""
    if n_sitios <= 0:
        return {}
    return {
        especie: n / n_sitios
        for especie, n in cuantizar_ocupacion(especies, fracciones, n_sitios)
    }


def especies_perdidas(
    especies: list[str],
    fracciones: Mapping[str, float],
    n_sitios: int,
) -> list[str]:
    """Especies que se pidieron con fracción > 0 y se quedan sin un solo átomo."""
    realizadas = dict(cuantizar_ocupacion(especies, fracciones, n_sitios))
    return [
        especie
        for especie in especies
        if float(fracciones.get(especie, 0.0)) > 0.0 and realizadas.get(especie, 0) == 0
    ]


def ajustar_fraccion(fraccion: float, n_sitios: int) -> float:
    """Lleva una fracción continua al múltiplo de `1/n_sitios` más cercano.

    Se acota a [1/n, 1-1/n]: si se pide una mezcla, el resultado debe tener al
    menos un átomo de cada especie. Devolver 0 convertiría la mezcla en un
    endmember con la etiqueta equivocada, que es justo lo que se quiere evitar.
    """
    if n_sitios <= 1:
        return float(fraccion)
    n = int(round(float(fraccion) * n_sitios))
    n = max(1, min(n_sitios - 1, n))
    return n / n_sitios


def clave_estructura(
    candidato: Any,
    supercell: Iterable[int],
    *,
    es_mixto: bool | None = None,
) -> tuple:
    """Identidad de la estructura que se construiría para este candidato.

    Dos candidatos con la misma clave producen los mismos átomos en la misma
    celda: calcular ambos por DFT es gastar el doble para obtener el mismo
    número. La fórmula NO sirve como identidad, porque `Cs0.95Rb0.051SnI3` y
    `CsSnI3` tienen fórmulas distintas y la misma estructura.
    """
    fracciones = getattr(candidato, "fractions", {}) or {}
    if es_mixto is None:
        es_mixto = any(
            len(getattr(candidato, f"{sitio}_site_species", []) or []) > 1
            for sitio in ("A", "B", "X")
        )
    sitios = sitios_por_subred(supercell if es_mixto else (1, 1, 1))

    partes: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for sitio in ("A", "B", "X"):
        especies = list(getattr(candidato, f"{sitio}_site_species", []) or [])
        cuentas = cuantizar_ocupacion(especies, fracciones.get(sitio, {}) or {}, sitios[sitio])
        # Ordenado: el reparto depende del orden de `especies`, pero dos listas
        # equivalentes con distinto orden describen la misma estructura.
        partes.append((sitio, tuple(sorted(c for c in cuentas if c[1] > 0))))
    return tuple(partes)
