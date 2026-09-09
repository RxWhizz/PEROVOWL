#!/usr/bin/env python3
"""Arranque de la interfaz gráfica del monitor DFT.

Pensado para ejecutarse sin argumentos y desde cualquier directorio:

    dft-monitor

Se encarga de lo que antes había que hacer a mano — crear la configuración,
elegir un `runs_dir` que exista, compilar el frontend la primera vez y abrir el
navegador — y deja el servidor en primer plano.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from . import paths

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def config_file() -> Path:
    return paths.config_file()


def example_config() -> Path:
    return paths.example_config()


def static_dir() -> Path:
    """SPA compilado: empaquetado en el binario, o donde lo deja npm."""
    empaquetado = paths.bundle_file("static")
    if empaquetado.is_dir():
        return empaquetado
    return paths.bundle_root() / "src" / "monitor_api" / "static"


def frontend_dir() -> Path:
    """Fuente del frontend. Solo existe ejecutando desde el repositorio."""
    return paths.bundle_root() / "frontend"

# Candidatos para el runs_dir inicial, en orden de preferencia. El primero que
# exista gana; `runs/` es un symlink a un volumen externo que puede no estar
# montado, y arrancar apuntando ahí muestra un panel vacío desconcertante.
RUNS_CANDIDATOS = (
    "runs/relax_basic",
    "local_runs/phase2_force/batch_000",
    "local_runs/phase2_force",
)

VERDE = "\033[32m"
AMARILLO = "\033[33m"
ROJO = "\033[31m"
GRIS = "\033[90m"
NEGRITA = "\033[1m"
FIN = "\033[0m"


def _color(texto: str, code: str) -> str:
    return f"{code}{texto}{FIN}" if sys.stdout.isatty() else texto


def di(texto: str = "") -> None:
    """print con flush.

    Con la salida redirigida, stdout es block-buffered y stderr no: el banner
    se quedaba en el buffer y los logs de la app aparecían antes.
    """
    print(texto, flush=True)


def _elegir_runs_dir() -> str:
    for rel in RUNS_CANDIDATOS:
        if paths.resolve_data(rel).is_dir():
            return rel
    return RUNS_CANDIDATOS[0]


def preparar_config(host: str, *, announce: bool = True) -> tuple[Path, str | None]:
    """Crea configs/monitor.yaml si falta. Devuelve (ruta, token generado)."""
    destino = config_file()
    if destino.exists():
        return destino, None

    ejemplo = example_config()
    if not ejemplo.exists():
        raise SystemExit(f"No se encuentra {ejemplo}. ¿Instalación incompleta?")

    texto = ejemplo.read_text(encoding="utf-8")

    runs_dir = _elegir_runs_dir()
    texto = texto.replace("  runs_dir: runs/relax_basic", f"  runs_dir: {runs_dir}", 1)

    # La app de escritorio se abre para *mirar*. Con auto_advance el poller
    # dispara el orquestador de active learning al detectar un lote acabado, así
    # que el primer doble clic sobre un batch terminado mutaría el pipeline sin
    # preguntar. Desde el repositorio se conserva `true`, que es lo que sostiene
    # la operación desatendida de siempre.
    if paths.is_frozen():
        texto = texto.replace(
            "  auto_advance: true",
            "  # Puesto a false al crear la config de la app de escritorio, que se\n"
            "  # abre para observar. Ponlo a true para operación desatendida.\n"
            "  auto_advance: false",
            1,
        )

    # Solo se genera token si de verdad hace falta: en localhost la auth
    # desactivada evita una pantalla de login que no protege de nada.
    token = None
    if not _es_local(host):
        token = secrets.token_urlsafe(32)
        texto = texto.replace('    token: ""', f'    token: "{token}"', 1)

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    if announce:
        di(_color(f"  Configuración creada: {destino}", VERDE))
        di(f"    runs_dir → {runs_dir}")
        if paths.is_frozen():
            di("    auto_advance → false (solo observa; edítalo para desatendido)")
    return destino, token


def _es_local(host: str) -> bool:
    return host.startswith("127.") or host in ("localhost", "::1")


def preparar_frontend(auto: bool) -> bool:
    """Compila el SPA si falta. Devuelve True si está disponible."""
    if (static_dir() / "index.html").is_file():
        return True

    frontend = frontend_dir()
    if paths.is_frozen() or not frontend.is_dir():
        # En un binario el SPA va dentro; si falta, es un fallo de empaquetado
        # y compilarlo no es una opción.
        return False

    if not auto:
        di(_color("  Frontend sin compilar — se sirve solo la API.", AMARILLO))
        di(f"    cd {frontend} && npm install && npm run build")
        return False

    npm = shutil.which("npm")
    if npm is None:
        di(_color("  Frontend sin compilar y npm no está instalado.", AMARILLO))
        di("    Instala Node.js (>=18) o usa solo la API.")
        return False

    if not (frontend / "node_modules").is_dir():
        di("  Instalando dependencias del frontend (solo la primera vez)…")
        if subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=frontend).returncode:
            di(_color("  Falló npm install.", ROJO))
            return False

    di("  Compilando el frontend…")
    if subprocess.run([npm, "run", "build"], cwd=frontend).returncode:
        di(_color("  Falló la compilación del frontend.", ROJO))
        return False

    return (static_dir() / "index.html").is_file()


def _esperar_servidor(url: str, timeout: float = 30.0) -> bool:
    """Espera a que el servidor responda."""
    limite = time.time() + timeout
    while time.time() < limite:
        try:
            with urllib.request.urlopen(f"{url}/auth/me", timeout=1):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


def abrir_navegador(url: str, timeout: float = 30.0) -> None:
    """Espera a que el servidor responda y abre el navegador. Nunca lanza."""
    if not _esperar_servidor(url, timeout=timeout):
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _puerto_efimero(host: str) -> int:
    """Reserva un puerto libre para el arranque local."""
    bind_host = "::1" if host == "::1" else host
    family = socket.AF_INET6 if bind_host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _abrir_shell(url: str, shell: str, timeout: float = 30.0) -> None:
    """Abre el shell visual solicitado. `app/webview` degradan a navegador si falta pywebview."""
    if shell in {"browser", "auto"}:
        abrir_navegador(url, timeout=timeout)
        return

    if not _esperar_servidor(url, timeout=timeout):
        return

    try:
        import webview  # type: ignore

        webview.create_window("Monitor DFT", url, width=1280, height=860)
        webview.start(gui="qt")
    except Exception:
        abrir_navegador(url, timeout=0.1)


def _servir_con_webview(destino: object, args: argparse.Namespace, recarga: bool, url: str) -> None:
    """Arranca uvicorn en segundo plano y deja pywebview en el hilo principal."""
    import uvicorn

    try:
        import webview  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "No se pudo abrir shell de escritorio: falta pywebview. "
            "Instala pywebview y un backend GUI como PyQt6."
        ) from exc

    if recarga:
        di(_color("  --reload no aplica con shell app/webview; se ignora.", AMARILLO))
        recarga = False

    config = uvicorn.Config(
        destino,
        host=args.host,
        port=args.port,
        reload=recarga,
        workers=1,
        log_level=args.log_level,
    )
    server = uvicorn.Server(config)
    errores: list[BaseException] = []

    def _run_server() -> None:
        try:
            server.run()
        except BaseException as exc:  # pragma: no cover - solo ruta GUI
            errores.append(exc)

    hilo = threading.Thread(target=_run_server, name="monitor-uvicorn", daemon=True)
    hilo.start()

    if not _esperar_servidor(url, timeout=30.0):
        server.should_exit = True
        hilo.join(timeout=5)
        if errores:
            raise SystemExit(f"No se pudo arrancar el servidor local: {errores[0]}") from errores[0]
        raise SystemExit(f"El servidor local no respondio en {url}")

    try:
        webview.create_window("Monitor DFT", url, width=1280, height=860)
        webview.start(gui="qt")
    except Exception as exc:
        server.should_exit = True
        hilo.join(timeout=5)
        raise SystemExit(f"No se pudo abrir la ventana local de escritorio: {exc}") from exc
    finally:
        server.should_exit = True
        hilo.join(timeout=10)


def _ready_payload(args: argparse.Namespace, url: str) -> dict[str, object]:
    """Contrato estable que consume la app Flutter para descubrir el motor."""
    return {
        "event": "ready",
        "base_url": url,
        "pid": os.getpid(),
        "data_root": str(paths.data_root()),
        "config_dir": str(paths.config_dir()),
        "frozen": paths.is_frozen(),
    }


def _servir_engine(destino: object, args: argparse.Namespace, recarga: bool, url: str) -> None:
    """Arranca el motor local para una GUI externa y avisa por stdout al estar listo."""
    import uvicorn

    if recarga:
        recarga = False

    config = uvicorn.Config(
        destino,
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
        log_level=args.log_level,
    )
    server = uvicorn.Server(config)
    errores: list[BaseException] = []

    def _run_server() -> None:
        try:
            server.run()
        except BaseException as exc:  # pragma: no cover - ruta de proceso
            errores.append(exc)

    hilo = threading.Thread(target=_run_server, name="monitor-engine", daemon=True)
    hilo.start()

    if not _esperar_servidor(url, timeout=30.0):
        server.should_exit = True
        hilo.join(timeout=5)
        if errores:
            raise SystemExit(f"No se pudo arrancar el motor local: {errores[0]}") from errores[0]
        raise SystemExit(f"El motor local no respondio en {url}")

    if args.print_ready_json:
        print(json.dumps(_ready_payload(args, url), ensure_ascii=False), flush=True)
    else:
        di(_color(f"  Motor local listo en {url}", VERDE))

    try:
        while hilo.is_alive():
            hilo.join(timeout=0.5)
            if errores:
                raise errores[0]
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        hilo.join(timeout=10)


def _avisar_runs_dir(cfg: dict) -> None:
    """Avisa si el runs_dir configurado no se puede leer.

    `runs/` y `calculations/` son symlinks a un volumen externo. Sin este aviso,
    el panel sale vacío y en calma — indistinguible de "no hay trabajo".
    """
    rel = (cfg.get("monitor") or {}).get("runs_dir", "runs/relax_basic")
    destino = paths.resolve_data(rel)
    if destino.is_dir():
        return

    di()
    di(_color(f"  Aviso: no se puede leer {rel}", AMARILLO))
    di(_color(f"    resuelve a {destino}", GRIS))
    di(_color("    ¿volumen externo sin montar? El monitor arranca igual.", GRIS))

    alternativa = next(
        (c for c in RUNS_CANDIDATOS if c != rel and paths.resolve_data(c).is_dir()), None
    )
    if alternativa:
        n = len([d for d in paths.resolve_data(alternativa).iterdir()
                 if (d / "status.json").exists()])
        di(_color(
            f"    Para probar ya: pon runs_dir: {alternativa} "
            f"en configs/monitor.yaml ({n} jobs)", GRIS,
        ))


def _correr_bucle_descubrimiento(args) -> None:
    """Ejecuta el bucle autónomo en primer plano de ESTE proceso.

    Lo invoca el servicio del monitor como subproceso desacoplado. Al ser un
    proceso propio, saturar la CPU aquí ya no afecta a la API: el servidor sigue
    respondiendo `/api/discovery/status` leyendo `state.json`, que este bucle va
    escribiendo.

    La salida va a stdout/stderr, que el servicio redirige a un fichero de log;
    si el bucle muere, ese log es lo único que explica por qué.
    """
    import traceback

    from . import paths
    from .services import discovery as servicio

    if args.data_root:
        os.environ["DFT_DATA_ROOT"] = str(Path(args.data_root).expanduser())

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    log = logging.getLogger("discovery.loop")

    try:
        bucle = servicio.build_loop()
        log.info("bucle arrancado en %s (pid %s)", paths.data_root(), os.getpid())
        bucle.run_forever(
            start_runner=not args.no_runner,
            dry_run=args.dry_run,
            use_mlff=False if args.no_mlff else None,
            max_rounds=args.max_rounds,
        )
        log.info("bucle terminado")
    except Exception:
        # Sin esto el traceback se perdería y el estado solo diría "murió".
        log.error("el bucle de descubrimiento falló:\n%s", traceback.format_exc())
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="dft-monitor",
        description="Interfaz gráfica del monitor DFT (GPAW/BUHO).",
    )
    ap.add_argument("--host", default="127.0.0.1",
                    help="Host de escucha (default 127.0.0.1). 0.0.0.0 lo expone en la red "
                         "y exige token de acceso.")
    ap.add_argument("--port", type=int, default=None,
                    help="Puerto. Si se omite en uso local, se elige uno libre.")
    ap.add_argument("--shell", default="auto", choices=["auto", "webview", "app", "browser"],
                    help="Shell visual (default auto). webview/app usan pywebview si está instalado.")
    ap.add_argument("--engine", action="store_true",
                    help="Modo motor embebido para una GUI externa: sin navegador ni banner humano.")
    ap.add_argument("--print-ready-json", action="store_true",
                    help="En modo --engine, imprime una línea JSON cuando el motor esté listo.")
    ap.add_argument("--data-root", metavar="DIR",
                    help="Raíz de los datos del proyecto (runs/, reports/, models/…). "
                         "Por defecto: DFT_DATA_ROOT, o el repositorio si se ejecuta "
                         "desde el código fuente, o el directorio actual.")
    ap.add_argument("--autodiagnostico", action="store_true",
                    help="Comprueba que este paquete lleva lo que necesita para "
                         "funcionar (tablas de calibracion, spglib, modelos), "
                         "imprime el informe en JSON y sale.")
    ap.add_argument("--no-browser", action="store_true", help="No abrir el navegador.")
    ap.add_argument("--no-build", action="store_true",
                    help="No compilar el frontend aunque falte.")
    ap.add_argument("--reload", action="store_true", help="Hot-reload (desarrollo).")
    ap.add_argument("--log-level", default="warning",
                    choices=["debug", "info", "warning", "error"])
    # El bucle de descubrimiento corre como PROCESO APARTE, no como hilo del
    # servidor: cribar decenas de miles de candidatos con pandas es CPU-bound y
    # con el GIL dejaba sin turno al event loop de asyncio — la API se quedaba
    # muda justo mientras había algo que monitorizar. Se relanza este mismo
    # ejecutable con esta bandera porque en el binario congelado `sys.executable`
    # es el propio binario y no existe un `python -m` al que llamar.
    ap.add_argument("--discovery-loop", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--max-rounds", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no-runner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-mlff", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.discovery_loop:
        _correr_bucle_descubrimiento(args)
        return

    # Antes que nada: el resto del arranque resuelve rutas contra esta raíz.
    if args.data_root:
        raiz = Path(args.data_root).expanduser()
        if not raiz.is_dir():
            raise SystemExit(f"--data-root no es un directorio: {raiz}")
        paths.set_data_root(raiz)

    if not paths.is_frozen():
        sys.path.insert(0, str(paths.bundle_root() / "src"))

    # Antes de levantar nada: preguntar por el paquete no necesita servidor, y
    # si algo critico falta es mejor decirlo aqui que arrancar y fallar luego en
    # silencio --- que es exactamente como se publicaron cuatro binarios sin
    # spglib.
    if args.autodiagnostico:
        from monitor_api.autodiagnostico import imprimir_y_salir

        raise SystemExit(imprimir_y_salir())

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            'Falta uvicorn. Instala las dependencias web:\n  pip install -e ".[web]"'
        ) from None

    machine_engine = bool(args.engine and args.print_ready_json)
    if args.engine:
        args.no_browser = True
        args.shell = "browser"
        if not _es_local(args.host):
            raise SystemExit("--engine solo escucha en localhost/127.0.0.1/::1")

    if not machine_engine:
        di()
        di(_color("  Monitor DFT", NEGRITA))
        di(_color(f"  datos:  {paths.data_root()}", GRIS))
        di(_color(f"  config: {paths.config_dir()}", GRIS))
        di()

    _, token_nuevo = preparar_config(args.host, announce=not machine_engine)
    hay_gui = False if args.engine else preparar_frontend(auto=not args.no_build)

    # Desde config.py y no desde main.py: ese módulo construye la app al
    # importarse y aquí solo hace falta leer la configuración.
    from monitor_api.config import load_config
    from monitor_api.security import load_auth_config

    cfg = load_config(config_file())

    try:
        auth = load_auth_config(cfg)
    except ValueError as exc:
        raise SystemExit(f"Error en la configuración: {exc}") from None

    if not _es_local(args.host) and not auth.enabled:
        raise SystemExit(
            f"--host {args.host} expone el monitor en la red sin autenticación.\n"
            "El monitor puede matar procesos y lanzar runners.\n\n"
            "Define monitor.auth.token en configs/monitor.yaml, o usa el default "
            "--host 127.0.0.1 para uso local."
        )

    if not machine_engine:
        _avisar_runs_dir(cfg)

    if args.port is None or args.port == 0:
        args.port = _puerto_efimero(args.host) if _es_local(args.host) else 8000

    from monitor_api.services.agent.config import load_agent_config
    from monitor_api.services.agent.runtime import ensure_managed_ollama

    agent_cfg = load_agent_config(cfg)
    if agent_cfg.enabled and agent_cfg.manage_service:
        if not machine_engine:
            di(_color("  Agente local: comprobando Ollama gestionado…", GRIS))
        # El agente es opcional; el monitor no. Antes un Ollama caído abortaba el
        # arranque entero con SystemExit, así que bastaba con reiniciar la
        # máquina —o descargar el binario en otro equipo— para que el monitor no
        # levantara. El gate duro sigue en pie en lo que importaba: no se cae a
        # CPU/Vulkan/llama.cpp. Simplemente el agente se queda sin servicio.
        motivo: str | None = None
        try:
            ok_agent = ensure_managed_ollama(agent_cfg, data_root=paths.data_root())
            if not ok_agent:
                motivo = "Ollama no respondió tras arrancar revive"
        except Exception as exc:
            motivo = str(exc)

        if motivo:
            aviso = (f"Agente local no disponible: {motivo}. "
                     f"El monitor arranca igual; el agente queda inactivo.")
            if machine_engine:
                print(aviso, file=sys.stderr, flush=True)
            else:
                di(_color(f"  {aviso}", AMARILLO))
        elif not machine_engine:
            di(_color(f"  Agente local: Ollama listo en {agent_cfg.base_url}", VERDE))

    url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"

    if not machine_engine:
        di()
        di(f"  {_color(url, VERDE)}")
        if not hay_gui:
            di(_color("  (solo API — el frontend no está compilado)", AMARILLO))
        if auth.enabled:
            if token_nuevo:
                di()
                di("  Token de acceso (guárdalo, no se vuelve a mostrar):")
                di(f"    {_color(token_nuevo, NEGRITA)}")
            else:
                di(_color("  Acceso con token — el de configs/monitor.yaml", GRIS))
        else:
            di(_color("  Sin autenticación (solo localhost)", GRIS))
        di()
        di(_color("  Ctrl+C para parar", GRIS))
        di()

    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    # Se configura el logging ANTES de que uvicorn importe la app: create_app()
    # llama a basicConfig, que es no-op si ya hay handlers, así que este nivel
    # gana y --log-level pasa a controlar también los logs de la aplicación.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    if paths.is_frozen():
        # Congelado no hay módulo que resolver por nombre: se le pasa el objeto.
        # `reload` deja de tener sentido y uvicorn lo rechazaría.
        from monitor_api.main import app as destino

        recarga = False
        if args.reload:
            di(_color("  --reload no aplica en el binario; se ignora.", AMARILLO))
    else:
        destino = "monitor_api.main:app"
        recarga = args.reload

    if args.engine:
        _servir_engine(destino, args, recarga, url)
        return

    if not args.no_browser and hay_gui and args.shell in {"app", "webview"}:
        _servir_con_webview(destino, args, recarga, url)
        di(_color("\n  Monitor detenido.\n", GRIS))
        return

    if not args.no_browser and hay_gui:
        threading.Thread(target=_abrir_shell, args=(url, args.shell), daemon=True).start()

    try:
        uvicorn.run(
            destino,
            host=args.host,
            port=args.port,
            reload=recarga,
            # workers=1 fijo: el estado del poller vive en memoria del proceso.
            workers=1,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        pass
    di(_color("\n  Monitor detenido.\n", GRIS))


if __name__ == "__main__":
    main()
