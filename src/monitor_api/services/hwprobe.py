"""Sondeo rápido de la máquina para dimensionar el reparto de trabajos DFT.

Esto **no** es la calibración. `scripts/bench_machine.py` mide el tiempo por
iteración lanzando GPAW de verdad y tarda ~15 min: eso encuentra el óptimo real.
Aquí se dimensiona en segundos a partir de recursos medidos, para que un botón
de arranque no tenga que pedir quince minutos antes de hacer nada.

El criterio físico del reparto:

- **La RAM manda.** Un job de GPAW sobre la supercelda 2x2x2 (40 atomos) ronda
  los 2 GB entre densidad, funciones de onda y el solapamiento del eigensolver.
  Si se abren mas slots de los que la RAM aguanta, el sistema pagina y el
  barrido entero se vuelve mas lento que con la mitad de slots.
- **Nucleos fisicos, no logicos.** GPAW en estas celdas esta limitado por ancho
  de banda de memoria, no por unidades enteras: el SMT reparte el mismo ancho de
  banda entre dos hilos y no aporta rendimiento.
- **Varios jobs pequenos rinden mas que uno grande.** La paralelizacion sobre
  puntos k y bandas satura pronto en celdas pequenas, asi que N jobs de C
  nucleos supera a 1 job de N*C.
- **Hay que dejar margen.** La API, la GUI y el sistema operativo necesitan
  nucleos y memoria; usar el 100 % hace que la maquina deje de responder.
"""
from __future__ import annotations

import logging
import platform
import time
from typing import Any

import psutil

log = logging.getLogger(__name__)

#: GB de RAM que se reservan por trabajo DFT concurrente. Conservador a
#: proposito: quedarse corto en slots cuesta tiempo, pasarse cuesta swap, y
#: paginar es mucho mas caro que un slot de menos.
RAM_POR_JOB_GB = 2.0

#: GB que se dejan para el sistema, la API y la GUI.
RAM_RESERVADA_SISTEMA_GB = 2.0

#: Nucleos que se dejan libres para que la maquina siga respondiendo.
NUCLEOS_RESERVADOS = 1

#: Techo de nucleos por trabajo. Mas alla, la paralelizacion de GPAW sobre una
#: celda de 40 atomos deja de escalar y solo anade sincronizacion.
MAX_NUCLEOS_POR_JOB = 4

#: Duracion objetivo del micro-benchmark de CPU, en segundos.
SEGUNDOS_BENCH = 1.5


def _micro_benchmark() -> dict[str, Any]:
    """GFLOP/s aproximados con multiplicacion de matrices densas.

    El recuento de nucleos miente en maquinas virtuales, contenedores con cuota
    y portatiles estrangulados por temperatura o por perfil de energia. Medir
    throughput real distingue "16 nucleos" de "16 nucleos que rinden como 4".
    """
    try:
        import numpy as np
    except ImportError:
        return {"disponible": False, "motivo": "numpy no disponible"}

    n = 512
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    flop_por_producto = 2.0 * n ** 3

    # Una pasada en frio para que BLAS levante sus hilos y no contaminar la medida.
    a @ b

    inicio = time.perf_counter()
    repeticiones = 0
    while time.perf_counter() - inicio < SEGUNDOS_BENCH:
        a @ b
        repeticiones += 1
    transcurrido = time.perf_counter() - inicio

    if transcurrido <= 0 or repeticiones == 0:
        return {"disponible": False, "motivo": "medicion demasiado corta"}
    return {
        "disponible": True,
        "gflops": round(repeticiones * flop_por_producto / transcurrido / 1e9, 2),
        "repeticiones": repeticiones,
        "segundos": round(transcurrido, 2),
    }


def medir(*, con_benchmark: bool = True) -> dict[str, Any]:
    """Recursos de la maquina, medidos ahora."""
    fisicos = psutil.cpu_count(logical=False)
    logicos = psutil.cpu_count(logical=True)
    # En algunos contenedores psutil no sabe distinguirlos.
    if not fisicos:
        fisicos = logicos or 1
    if not logicos:
        logicos = fisicos

    try:
        freq = psutil.cpu_freq()
    except (OSError, NotImplementedError, AttributeError):
        freq = None

    mem = psutil.virtual_memory()
    datos: dict[str, Any] = {
        "os": platform.system(),
        "cpu": platform.processor() or platform.machine(),
        "nucleos_fisicos": int(fisicos),
        "nucleos_logicos": int(logicos),
        "smt": int(logicos) > int(fisicos),
        "frecuencia_mhz": round(freq.current, 0) if freq and freq.current else None,
        "frecuencia_max_mhz": round(freq.max, 0) if freq and freq.max else None,
        "ram_total_gb": round(mem.total / 1e9, 2),
        "ram_disponible_gb": round(mem.available / 1e9, 2),
        "carga_cpu_pct": psutil.cpu_percent(interval=0.3),
    }
    datos["benchmark"] = _micro_benchmark() if con_benchmark else {"disponible": False,
                                                                   "motivo": "omitido"}
    return datos


def repartir(medicion: dict[str, Any]) -> dict[str, Any]:
    """Cuantos trabajos concurrentes y cuantos nucleos por trabajo.

    Devuelve tambien `limitado_por`, porque un reparto sin explicacion no se
    puede discutir: si salen dos slots en una maquina de 16 nucleos, hay que
    poder ver que fue la RAM y no un error.
    """
    fisicos = int(medicion.get("nucleos_fisicos") or 1)
    ram_disp = float(medicion.get("ram_disponible_gb") or 0.0)

    nucleos_utiles = max(1, fisicos - NUCLEOS_RESERVADOS)
    ram_utilizable = max(0.0, ram_disp - RAM_RESERVADA_SISTEMA_GB)

    techo_ram = int(ram_utilizable // RAM_POR_JOB_GB)
    techo_cpu = nucleos_utiles  # como minimo un nucleo por slot

    slots = max(1, min(techo_ram, techo_cpu))
    nucleos_por_job = max(1, min(MAX_NUCLEOS_POR_JOB, nucleos_utiles // slots))

    if techo_ram < 1:
        limitado_por = "ram"
        aviso = (
            f"Solo hay {ram_disp:.1f} GB de RAM disponibles. Se abre un unico "
            f"trabajo y aun asi puede paginar: cada job DFT pide ~{RAM_POR_JOB_GB:.0f} GB."
        )
    elif techo_ram <= techo_cpu:
        limitado_por = "ram"
        aviso = None
    else:
        limitado_por = "cpu"
        aviso = None

    return {
        "runner_slots": int(slots),
        "runner_cores": int(nucleos_por_job),
        "nucleos_en_uso": int(slots * nucleos_por_job),
        "nucleos_utiles": int(nucleos_utiles),
        "ram_comprometida_gb": round(slots * RAM_POR_JOB_GB, 1),
        "limitado_por": limitado_por,
        "aviso": aviso,
        "supuestos": {
            "ram_por_job_gb": RAM_POR_JOB_GB,
            "ram_reservada_sistema_gb": RAM_RESERVADA_SISTEMA_GB,
            "nucleos_reservados": NUCLEOS_RESERVADOS,
            "max_nucleos_por_job": MAX_NUCLEOS_POR_JOB,
        },
    }


def sondear(*, con_benchmark: bool = True) -> dict[str, Any]:
    """Mide y reparte de una vez."""
    medicion = medir(con_benchmark=con_benchmark)
    return {
        "maquina": medicion,
        "reparto": repartir(medicion),
        # Que quede claro que esto dimensiona, no calibra: `bench_machine.py`
        # mide t/iteracion con GPAW real y es lo que encuentra el optimo.
        "es_estimacion": True,
        "refinar_con": "scripts/bench_machine.py --quick",
    }
