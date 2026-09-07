"""FastAPI app — DFT Simulation Monitor."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from . import __version__, paths
from .config import load_config
from .metrics_history import MetricsHistory
from .poller import DFTPoller
from .router import router
from .security import install_auth, load_auth_config
from .security import router as auth_router
from .services.agent import router as agent_router
from .telegram_bot import run_listener
from .ws import EventHub
from .ws import router as ws_router

log = logging.getLogger(__name__)

# El SPA compilado viaja con el programa: en el binario lo extrae PyInstaller,
# desde el repositorio está donde lo deja `npm run build`.


def _static_dir() -> Path:
    empaquetado = paths.bundle_file("static")
    if empaquetado.is_dir():
        return empaquetado
    return Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = app.state.config
    monitor_cfg = cfg.get("monitor", {})
    telegram_cfg = cfg.get("telegram", {})

    # Resolver runs_dir relativo al root del proyecto
    runs_rel = monitor_cfg.get("runs_dir", "runs/relax_basic")
    runs_dir = paths.resolve_data(runs_rel).resolve()

    poller = DFTPoller(
        runs_dir=runs_dir,
        cfg=monitor_cfg,
        telegram_cfg=telegram_cfg,
    )
    app.state.poller = poller

    if not runs_dir.is_dir():
        # runs/ y calculations/ son symlinks a un volumen externo. Si no está
        # montado, el poller no vería jobs y todo parecería "en calma".
        log.warning(
            "runs_dir NO disponible: %s — ¿volumen externo sin montar? "
            "El monitor arranca igualmente; consulta GET /api/health.",
            runs_dir,
        )

    hub = EventHub()
    app.state.hub = hub
    hub.start(poller)

    metrics_history = MetricsHistory()
    app.state.metrics_history = metrics_history
    metrics_task = asyncio.create_task(
        metrics_history.run_forever(monitor_cfg.get("metrics_interval_sec", 10))
    )

    interval = monitor_cfg.get("poll_interval_sec", 30)
    poller_task = asyncio.create_task(poller.run_forever(interval))

    token   = telegram_cfg.get("bot_token", "")
    chat_id = telegram_cfg.get("chat_id", "")
    bot_task = asyncio.create_task(run_listener(token, chat_id, poller))

    log.info("Monitor DFT iniciado — runs_dir=%s", runs_dir)

    yield

    await hub.stop()
    poller_task.cancel()
    bot_task.cancel()
    metrics_task.cancel()
    for t in (poller_task, bot_task, metrics_task):
        try:
            await t
        except asyncio.CancelledError:
            pass


# Prefijos que pertenecen al backend: un 404 ahí es un 404, no el shell del SPA.
_API_PREFIXES = ("/api/", "/auth/", "/ws/", "/openapi.json", "/docs", "/redoc")


class _SPAStaticFiles(StaticFiles):
    """Sirve el SPA: cualquier ruta desconocida cae en index.html.

    El enrutado del frontend es de cliente, así que recargar en /jobs debe
    devolver el shell. Las rutas del backend quedan excluidas: devolver HTML
    con 200 ante un endpoint mal escrito enmascararía el error al cliente.
    """

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # StaticFiles LANZA 404 en vez de devolverlo, así que no basta con
            # mirar response.status_code.
            if exc.status_code != 404 or self._es_ruta_de_api(scope):
                raise
            return await super().get_response("index.html", scope)

        if response.status_code == 404 and not self._es_ruta_de_api(scope):
            return await super().get_response("index.html", scope)
        return response

    @staticmethod
    def _es_ruta_de_api(scope) -> bool:
        return scope.get("path", "").startswith(_API_PREFIXES)


def _mount_frontend(app: FastAPI) -> None:
    """Monta el SPA compilado. Va AL FINAL para que /api y /ws ganen."""
    if not (_static_dir() / "index.html").is_file():
        log.info(
            "Sin frontend compilado en %s — la API funciona igual. "
            "Compílalo con: cd frontend && npm install && npm run build",
            _static_dir(),
        )
        return
    app.mount("/", _SPAStaticFiles(directory=_static_dir(), html=True), name="frontend")
    log.info("Frontend servido desde %s", _static_dir())


def create_app(config: dict | None = None) -> FastAPI:
    """Construye la app.

    `config` permite inyectar la configuración en vez de leer
    `configs/monitor.yaml`. Los tests la usan para no depender de si esa ruta
    existe en la máquina donde corren.
    """
    # basicConfig no hace nada si el logger raíz ya tiene handlers, así que
    # quien arranque la app (p. ej. el lanzador) puede fijar antes su nivel.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # La config se resuelve aquí y no en el lifespan porque el middleware de
    # auth depende de ella y Starlette no admite añadir middleware una vez
    # arrancada la app.
    cfg = load_config() if config is None else config

    # El YAML puede fijar la raíz de datos, pero cede ante --data-root y ante
    # DFT_DATA_ROOT (de eso se encarga set_data_root con override=False).
    declarada = (cfg.get("monitor") or {}).get("data_root")
    if declarada:
        paths.set_data_root(declarada, override=False)

    # Con la raiz de datos ya resuelta: deja el pipeline en fuente donde el
    # runner externo (y WSL) puedan leerlo. Es idempotente y solo copia cuando
    # cambia la version, asi que no cuesta nada en arranques normales.
    paths.materializar_pipeline(version=__version__)

    app = FastAPI(
        title="DFT Simulation Monitor",
        description="Monitor en tiempo real de simulaciones GPAW/BUHO con push via Telegram",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.config = cfg

    install_auth(app, load_auth_config(cfg))

    app.include_router(auth_router)
    app.include_router(router)
    app.include_router(agent_router)
    app.include_router(ws_router)

    _mount_frontend(app)
    return app


app = create_app()
