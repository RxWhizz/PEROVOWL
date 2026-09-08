"""Arranque de un clic: dimensionar la maquina y lanzar el protocolo autonomo.

Encadena lo que hasta ahora habia que hacer a mano y en orden — medir la
maquina, fijar el reparto de trabajos, comprobar que estan las piezas, enumerar
el espacio quimico y arrancar el bucle— y deja constancia de cada paso.

Por que se comprueban los prerequisitos ANTES de arrancar: el bucle es un
subproceso desacoplado que corre durante dias. Si falta GPAW o los datasets PAW,
sin esta comprobacion la primera ronda criba, selecciona, prepara los trabajos y
solo entonces falla, con el estado a medias y el usuario mirando una barra que
no avanza. Es mas barato decir que falta antes de empezar.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import yaml

from .. import paths
from . import hwprobe

log = logging.getLogger(__name__)

_lock = threading.RLock()
_ultimo: dict[str, Any] = {}


def _paso(nombre: str, ok: bool, detalle: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"paso": nombre, "ok": ok, "detalle": detalle, "error": error}


def _persistir_reparto(reparto: dict[str, Any]) -> dict[str, Any]:
    """Escribe runner_slots/runner_cores en monitor.yaml.

    Se conserva el resto del fichero: es configuracion del usuario, no un
    documento que este servicio pueda reescribir entero.
    """
    ruta = paths.config_file()
    cfg: dict[str, Any] = {}
    if ruta.is_file():
        try:
            with ruta.open(encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f"No se pudo leer {ruta}: {exc}") from exc

    # Bajo `monitor:`, que es la seccion que el poller recibe como su cfg
    # (`main.py` le pasa `cfg["monitor"]`). Escrito al nivel superior el fichero
    # queda valido y nadie lo lee: el arranque diria "configuracion ok" y
    # seguiria usando los slots de antes.
    seccion = cfg.get("monitor")
    if not isinstance(seccion, dict):
        seccion = {}
        cfg["monitor"] = seccion

    previo = {
        "runner_slots": seccion.get("runner_slots"),
        "runner_cores": seccion.get("runner_cores"),
    }
    seccion["runner_slots"] = int(reparto["runner_slots"])
    seccion["runner_cores"] = int(reparto["runner_cores"])

    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Atomica: si el proceso muere a medias, la configuracion anterior sigue
    # siendo valida en vez de quedar un YAML truncado que no arranca.
    parcial = ruta.with_suffix(".yaml.parcial")
    with parcial.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    parcial.replace(ruta)
    return {
        "anterior": previo,
        "nuevo": {k: seccion[k] for k in ("runner_slots", "runner_cores")},
        "fichero": str(ruta),
    }


def _aplicar_en_caliente(poller: Any, reparto: dict[str, Any]) -> bool:
    """Mete el reparto en la configuracion viva del poller.

    `poller.cfg` es la seccion `monitor:` que `main.py` le paso al arrancar, y
    es de donde salen `runner_slots`/`runner_cores` cuando se lanza un runner.
    Sin esto habria que reiniciar la app para que la medicion sirviera de algo.
    """
    cfg = getattr(poller, "cfg", None)
    if not isinstance(cfg, dict):
        return False
    cfg["runner_slots"] = int(reparto["runner_slots"])
    cfg["runner_cores"] = int(reparto["runner_cores"])
    return True


def autoconfigurar_dft() -> dict[str, Any]:
    """Busca un GPAW ya instalado en WSL y apunta la configuracion hacia el.

    Que no este configurado no significa que no este instalado. La comprobacion
    de capacidad decia "Define discovery.wsl.python en config/generator.yaml" en
    cuanto faltaba la ruta, y en una maquina donde GPAW ya vivia en WSL lo unico
    que hacia falta era mirar. Pedirle a alguien que edite un YAML para algo que
    la maquina puede averiguar sola no tiene defensa.

    Solo LEE: si no encuentra nada, no instala --- eso son ~2.5 GB y se ofrece
    aparte, no se hace a escondidas.
    """
    from buho import setup_wizard

    from . import setup

    try:
        sondeo = setup_wizard.sondear_gpaw_wsl()
    except Exception as exc:  # noqa: BLE001 - detectar no puede tumbar el arranque
        log.warning("no se pudo buscar GPAW en WSL: %s", exc)
        return {"configurado": False, "motivo": f"{type(exc).__name__}: {exc}"}
    hallado = sondeo.get("hallado")
    if not hallado:
        return {"configurado": False, "sondeo": sondeo,
                "motivo": sondeo.get("motivo") or "no hay ningun GPAW instalado en WSL"}

    # Se le pasan las rutas HALLADAS: sintetizar las canonicas dejaria la
    # config apuntando a un entorno que no existe si el GPAW de la maquina vive
    # en otro sitio (miniconda3, mambaforge, /opt/conda...).
    escrito = setup.configurar_wsl_dft(distro=hallado.get("distro"), rutas=hallado)
    return {"configurado": bool(escrito.get("escrito")), "detectado": hallado,
            "sondeo": sondeo, "escritura": escrito}


#: Que decirle a alguien cuando la busqueda no encontro GPAW. El mensaje de la
#: capacidad ("Define discovery.wsl.python en config/generator.yaml") es correcto
#: solo mientras nadie haya mirado; despues de mirar es un consejo para otro
#: problema --- mandar a editar un YAML a quien le faltan 2.5 GB de descarga.
def _explicar_sondeo(sondeo: dict[str, Any]) -> tuple[str, str]:
    if not sondeo.get("wsl"):
        return ("No hay WSL en esta máquina.",
                "Instala WSL con 'wsl --install' y reinicia; luego instala el "
                "runtime DFT desde la pestaña Entorno.")
    if sondeo.get("hallado"):
        return ("", "")
    if not sondeo.get("distros"):
        # `wsl.exe` existe en todo Windows 11 aunque no haya nada dentro.
        return ("Windows trae wsl.exe, pero no hay ninguna distribución de Linux "
                "instalada dentro de WSL.",
                "Abre PowerShell como administrador, ejecuta 'wsl --install -d "
                "Ubuntu', reinicia, y después instala el runtime DFT desde la "
                "pestaña Entorno.")
    candidatos = sondeo.get("candidatos") or []
    if not candidatos:
        return ("Hay WSL, pero la distro no respondió o no tiene ningún Python.",
                "Ábrela una vez (menú Inicio → Ubuntu) para que termine de "
                "configurarse, y vuelve a intentarlo.")
    return (f"Busqué GPAW en WSL y no está: probé {len(candidatos)} intérprete(s) "
            f"y ninguno lo tiene.",
            "Instálalo desde la pestaña Entorno (~2.5 GB de descarga).")


def _faltantes(*, autoconfigurar: bool = True) -> list[dict[str, Any]]:
    """Capacidades requeridas que no estan listas.

    Antes de dar por perdido el runtime DFT se intenta encontrarlo: es gratis y
    resuelve el caso comun de "instalado pero sin configurar".
    """
    from . import setup

    datos = setup.status(fast=True)
    sondeo: dict[str, Any] | None = None
    if autoconfigurar and any(
        c.get("id") == "dft" and c.get("requerido") and not c.get("ok")
        for c in (datos.get("capacidades") or [])
    ):
        intento = autoconfigurar_dft()
        if intento.get("configurado"):
            datos = setup.status(fast=True)
        else:
            sondeo = intento.get("sondeo")
    faltan = []
    for cap in datos.get("capacidades", []) or []:
        if cap.get("requerido") and not cap.get("ok"):
            entrada = {
                "id": cap.get("id"),
                "titulo": cap.get("titulo"),
                "error": cap.get("error"),
                "remediacion": cap.get("remediacion"),
            }
            if cap.get("id") == "dft" and sondeo is not None:
                # Ya se busco: contar lo que se vio en vez de repetir el consejo
                # de "define la ruta", que solo vale mientras nadie ha mirado.
                error, remediacion = _explicar_sondeo(sondeo)
                if error:
                    entrada["error"] = error
                    entrada["remediacion"] = remediacion
                    entrada["sondeo"] = sondeo
            faltan.append(entrada)
    return faltan


def _calidad_del_surrogate() -> dict[str, Any] | None:
    """Ultima metrica de reentrenamiento, para no arrancar a ciegas.

    El bucle gasta dias de DFT guiado por el surrogate. Si `cv_mae` no mejora
    sobre el `baseline` de predecir la media, ese gasto se reparte casi al azar.
    No se bloquea el arranque —al principio, sin datos, es lo esperable— pero se
    dice, porque es la diferencia entre explorar y tirar computo.
    """
    ruta = paths.resolve_data("data/discovery/model_metrics.jsonl")
    if not ruta.is_file():
        return None
    ultima = None
    try:
        with ruta.open(encoding="utf-8") as fh:
            for linea in fh:
                linea = linea.strip()
                if linea:
                    try:
                        ultima = json.loads(linea)
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return None
    if not isinstance(ultima, dict):
        return None

    cv = ultima.get("cv_mae_eV")
    base = ultima.get("baseline_mae_eV")
    mejora = None
    if isinstance(cv, (int, float)) and isinstance(base, (int, float)) and base > 0:
        mejora = round(100.0 * (1.0 - cv / base), 1)
    return {
        "round_id": ultima.get("round_id"),
        "n_samples": ultima.get("n_samples"),
        "cv_mae_eV": cv,
        "baseline_mae_eV": base,
        "mejora_sobre_media_pct": mejora,
    }


def estado() -> dict[str, Any]:
    """Lo que dejo el ultimo arranque, mas el estado actual del protocolo."""
    from . import discovery

    with _lock:
        ultimo = dict(_ultimo)
    ultimo["protocolo"] = discovery.status()
    if "surrogate" not in ultimo:
        ultimo["surrogate"] = _calidad_del_surrogate()
    return ultimo


def arrancar(
    *,
    poller: Any = None,
    max_rounds: int | None = None,
    use_mlff: bool | None = None,
    dry_run: bool = False,
    con_benchmark: bool = True,
) -> dict[str, Any]:
    """Mide, configura, comprueba y lanza. Devuelve el detalle de cada paso."""
    global _ultimo
    pasos: list[dict[str, Any]] = []
    inicio = time.time()

    def _terminar(ok: bool, motivo: str | None, **extra: Any) -> dict[str, Any]:
        global _ultimo
        resultado = {
            "ok": ok,
            "arrancado": ok,
            "motivo": motivo,
            "pasos": pasos,
            "surrogate": _calidad_del_surrogate(),
            "segundos": round(time.time() - inicio, 1),
            **extra,
        }
        with _lock:
            _ultimo = resultado
        return resultado

    # 1 - Medir la maquina
    try:
        sondeo = hwprobe.sondear(con_benchmark=con_benchmark)
        pasos.append(_paso("sondeo", True, sondeo))
    except Exception as exc:  # un sondeo fallido no debe impedir arrancar
        log.warning("sondeo de hardware fallido: %s", exc)
        sondeo = None
        pasos.append(_paso("sondeo", False, error=f"{type(exc).__name__}: {exc}"))

    # 2 - Fijar el reparto
    if sondeo is not None:
        try:
            detalle = _persistir_reparto(sondeo["reparto"])
            # Escribirlo no basta: la configuracion se lee al arrancar el
            # proceso, asi que el runner que este arranque lance seguiria usando
            # los valores con los que se abrio la app. Medir 19 slots y correr
            # con 2 es exactamente lo contrario de lo que promete el boton.
            detalle["aplicado_en_caliente"] = _aplicar_en_caliente(
                poller, sondeo["reparto"])
            pasos.append(_paso("configuracion", True, detalle))
        except (OSError, RuntimeError) as exc:
            pasos.append(_paso("configuracion", False, error=str(exc)))

    # 3 - Prerequisitos: aqui se para, antes de gastar nada
    faltan = _faltantes()
    pasos.append(_paso("prerequisitos", not faltan, {"faltantes": faltan}))
    if faltan:
        return _terminar(False, "faltan requisitos")

    from . import discovery

    # 4 - Enumerar el espacio quimico si aun no existe
    try:
        previo = (discovery.status().get("state", {}) or {}).get("status")
        if previo in (None, "not_initialized"):
            pasos.append(_paso("init", True, discovery.init(reset=False)))
        else:
            pasos.append(_paso("init", True, {"omitido": "el espacio ya estaba enumerado"}))
    except Exception as exc:
        pasos.append(_paso("init", False, error=f"{type(exc).__name__}: {exc}"))
        return _terminar(False, "no se pudo enumerar el espacio")

    # 5 - Arrancar el bucle
    try:
        lanzado = discovery.start(
            start_runner=True, dry_run=dry_run, use_mlff=use_mlff, max_rounds=max_rounds,
        )
        pasos.append(_paso("protocolo", True, {"max_rounds": max_rounds, "dry_run": dry_run}))
    except Exception as exc:
        pasos.append(_paso("protocolo", False, error=f"{type(exc).__name__}: {exc}"))
        return _terminar(False, "no se pudo arrancar el protocolo")

    return _terminar(
        True, None,
        reparto=(sondeo or {}).get("reparto"),
        protocolo=lanzado,
    )


def reset_for_tests() -> None:
    global _ultimo
    with _lock:
        _ultimo = {}
