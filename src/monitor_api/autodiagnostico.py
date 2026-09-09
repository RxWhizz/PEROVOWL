"""Que viaja de verdad en este binario.

Por que existe
--------------
Las siete ultimas versiones se rompieron todas de la misma forma: un fichero o
un modulo que el codigo lee en ejecucion y que no llegaba al paquete. spglib se
publico ausente cuatro veces seguidas --- la identificacion de fase daba "no
disponible" y nadie se enteraba--- y `config/eg_scale.json` no se empaquetaba,
asi que en todo binario publicado el desplazamiento era 0 y el Tier 1 descartaba
el 100 % de los candidatos. En ambos casos el arreglo estaba en el codigo y en
el repositorio: arreglado en el codigo y sin empaquetar es lo mismo que no
arreglado.

Ninguna prueba podia verlo, porque las pruebas corren sobre el arbol de fuentes
y ahi todo esta. Esto lo pregunta al artefacto: una sola orden que el binario
responde sobre si mismo, en JSON, y que sale con codigo 1 si falta algo critico.
La corre CI sobre el paquete ya comprimido y reextraido, y la puede correr
cualquiera que reporte un problema.

Lo que se comprueba no es que un modulo importe, sino que **haga su trabajo**:
spglib tiene que identificar una perovskita cubica como Pm-3m, no solo importar.
"""
from __future__ import annotations

import json
from typing import Any

from . import paths

#: Parametro de red de la celda de prueba. Cualquiera vale: lo que se comprueba
#: es la simetria, que no depende del tamano.
_A_PRUEBA = 6.0

#: Modelos de bandgap, en el mismo orden de preferencia que usa la cascada: el
#: reentrenado gana al de fabrica.
_SURROGATES = (
    ("models", "discovery", "surrogate_bandgap_current.pkl"),
    ("models", "surrogate_bandgap.pkl"),
)


def _cubica_de_prueba():
    """Una ABX3 ideal en Pm-3m, construida a mano.

    A mano y no con el generador a proposito: aqui se esta preguntando por
    spglib, y meter el constructor de estructuras por medio haria que un fallo
    suyo se leyera como un fallo de simetria.
    """
    from ase import Atoms

    return Atoms(
        symbols=["Cs", "Pb", "I", "I", "I"],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ],
        cell=[_A_PRUEBA] * 3,
        pbc=True,
    )


def _comprobar_simetria() -> dict[str, Any]:
    """spglib identificando, no spglib importando.

    En el binario de 0.7.0 a 0.7.3 el modulo no viajaba: `grupo_espacial`
    devolvia {"disponible": False} y el pipeline seguia como si nada, con la
    fase sin identificar. Un `import spglib` habria fallado igual de fuerte,
    pero esto ademas cubre que la version instalada siga hablando el mismo
    dialecto (spglib nuevo devuelve un dataclass; el viejo, un dict).
    """
    try:
        from buho.structure.fases import grupo_espacial

        r = grupo_espacial(_cubica_de_prueba())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if not r.get("disponible"):
        return {"ok": False, "error": r.get("motivo") or "spglib no disponible"}
    if int(r.get("numero", 0)) != 221:
        return {"ok": False, "error": f"una cubica ideal salio {r.get('simbolo')} "
                                      f"({r.get('numero')}), no Pm-3m (221)"}
    return {"ok": True, "simbolo": r.get("simbolo"), "numero": r.get("numero")}


def _comprobar_tablas() -> dict[str, Any]:
    """Las dos mitades de la correccion de bandgap.

    Tienen que estar las dos y salir de la misma corrida: la etiqueta lleva el
    scissor de SOC y la prediccion lleva la escala, y calibradas a mallas k
    distintas no componen --- medido, 13.6 % de error real frente al 4.8 % que
    anunciaba la tabla.
    """
    from buho import bandgap_scissor, eg_scale

    escala = eg_scale.describir()
    scissor = bandgap_scissor.describir()
    ok = bool(escala.get("disponible")) and bool(scissor.get("disponible"))
    salida: dict[str, Any] = {"ok": ok, "escala": escala, "scissor": scissor}
    if not ok:
        salida["error"] = (
            "sin tabla de escala: el bandgap no se lleva a escala experimental "
            "y el Tier 1 descarta contra una magnitud que no es la suya"
            if not escala.get("disponible") else
            "sin tabla de scissor: las etiquetas de entrenamiento no llevan SOC")
    return salida


def _comprobar_surrogate() -> dict[str, Any]:
    """Que el modelo cargue Y declare la escala en la que se entreno.

    Las dos cosas han fallado por separado: un modelo que no carga por la
    version de scikit-learn deja el cribado sin Tier 1, y uno sin sello hace que
    la correccion no se aplique --- que era el fallo con mejor aspecto de todos,
    porque el cribado seguia produciendo numeros plausibles.
    """
    from buho import eg_scale

    for rel in _SURROGATES:
        ruta = paths.find_resource(*rel)
        if not ruta.exists():
            continue
        try:
            from ml_surrogate.model import SurrogateEnsemble

            modelo = SurrogateEnsemble.load(ruta)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "ruta": str(ruta),
                    "error": f"no carga ({type(exc).__name__}: {exc})"}
        sello = eg_scale.escala_del_modelo(modelo)
        return {
            "ok": True,
            "ruta": str(ruta),
            "escala": sello,
            "corrige": bool(eg_scale.aplicable_a(modelo)),
            # Sin sello no es un fallo del paquete: el modelo de fabrica se
            # entreno en otra escala y la correccion entra tras el primer
            # reentrenamiento. Pero tiene que verse.
            "aviso": None if sello else (
                "el modelo no declara escala: hasta el primer reentrenamiento "
                "el cribado usa el valor calculado sin corregir"),
        }
    return {"ok": False, "error": "ningun modelo de bandgap en el paquete ni en los datos"}


def _comprobar_estructuras() -> dict[str, Any]:
    ruta = paths.find_resource("structures")
    if not ruta.is_dir():
        return {"ok": False, "error": f"sin structures/ en {ruta}"}
    n = sum(1 for _ in ruta.glob("*"))
    return {"ok": n > 0, "ruta": str(ruta), "n": n,
            **({} if n else {"error": "structures/ existe pero esta vacio"})}


def diagnosticar() -> dict[str, Any]:
    """El informe completo. `ok` es falso si algo critico falta."""
    from . import __version__

    comprobaciones = {
        "tablas_calibracion": _comprobar_tablas(),
        "simetria": _comprobar_simetria(),
        "surrogate": _comprobar_surrogate(),
        "estructuras": _comprobar_estructuras(),
    }
    fallos = [
        f"{nombre}: {r.get('error', 'fallo sin detalle')}"
        for nombre, r in comprobaciones.items() if not r.get("ok")
    ]
    avisos = [f"{nombre}: {r['aviso']}"
              for nombre, r in comprobaciones.items() if r.get("aviso")]
    return {
        "ok": not fallos,
        "version": __version__,
        "rutas": paths.describe(),
        "comprobaciones": comprobaciones,
        "fallos": fallos,
        "avisos": avisos,
    }


def imprimir_y_salir() -> int:
    """Una linea JSON por stdout y el codigo de salida. Para CI y para soporte."""
    try:
        informe = diagnosticar()
    except Exception as exc:  # noqa: BLE001 - la respuesta SIEMPRE es JSON
        import traceback

        informe = {"ok": False, "fallos": [f"{type(exc).__name__}: {exc}"],
                   "traceback": traceback.format_exc(limit=6)}
    print(json.dumps(informe, ensure_ascii=False, default=str))
    return 0 if informe.get("ok") else 1
