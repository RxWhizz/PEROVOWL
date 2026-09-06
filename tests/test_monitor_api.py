"""Tests de la API del monitor DFT (src/monitor_api).

Cubren específicamente los fallos detectados al planificar la GUI:
enmascaramiento de rutas, volumen externo desmontado y paginación.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

fastapi = pytest.importorskip("fastapi", reason="requiere el extra [web]")
from fastapi.testclient import TestClient  # noqa: E402

from monitor_api.main import create_app  # noqa: E402
from monitor_api.models import StatsResponse  # noqa: E402


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Aísla los tests del entorno del desarrollador.

    DFT_MONITOR_TOKEN tiene prioridad sobre la configuración, así que tenerla
    exportada en la shell activaba la autenticación y hacía fallar todo el
    módulo con 401.
    """
    monkeypatch.delenv("DFT_MONITOR_TOKEN", raising=False)
    monkeypatch.delenv("DFT_MONITOR_SESSION_SECRET", raising=False)
    monkeypatch.delenv("DFT_DATA_ROOT", raising=False)
    monkeypatch.delenv("DFT_MONITOR_CONFIG_DIR", raising=False)

    # La raíz de datos es estado de módulo: sin resetear, un test la fijaría
    # para todos los siguientes.
    from monitor_api import paths as _paths

    _paths.reset_data_root()
    yield
    _paths.reset_data_root()


class StubPoller:
    """Poller mínimo: los endpoints solo necesitan runs_dir, cfg y snapshots."""

    def __init__(self, runs_dir: Path, snapshots=None, cfg=None):
        self.runs_dir = runs_dir
        self.cfg = cfg or {}
        self._snapshots = {s.job_id: s for s in (snapshots or [])}
        self.last_poll_at = time.time()

    @property
    def snapshots(self):
        return self._snapshots


def _snap(job_id: str, formula: str, status: str, **kw) -> StatsResponse:
    return StatsResponse(job_id=job_id, formula=formula, status=status, **kw)


SNAPSHOTS = [
    _snap("aaa1", "CsPbI3",   "converged", elapsed_min=10.0, final_energy_ev=-12.5),
    _snap("bbb2", "MASnI3",   "converged", elapsed_min=30.0, final_energy_ev=-20.1),
    _snap("ccc3", "FAPbBr3",  "running",   elapsed_min=5.0),
    _snap("ddd4", "CsSnI3",   "failed",    elapsed_min=1.0),
    _snap("eee5", "FASnBr3",  "pending"),
]


@pytest.fixture
def client(tmp_path):
    """App con un poller determinista y runs_dir existente.

    Se instancia TestClient sin `with`, así el lifespan real (poller de fondo +
    listener de Telegram) no arranca y el test no depende del disco.
    """
    app = create_app(config={})
    app.state.poller = StubPoller(tmp_path, SNAPSHOTS)
    app.state.hub = None
    return TestClient(app)


# ── Regresión: ruta enmascarada ──────────────────────────────────────────────

def test_converged_no_lo_captura_la_ruta_parametrica(client):
    """`/api/jobs/converged` se declaraba tras `/api/jobs/{job_id}`.

    Starlette resuelve por orden de registro, así que caía en el handler
    paramétrico con job_id="converged" y devolvía 404 siempre.
    """
    r = client.get("/api/jobs/converged")
    assert r.status_code == 200, "la ruta paramétrica volvió a ensombrecer /converged"

    body = r.json()
    assert isinstance(body, list)
    assert {j["formula"] for j in body} == {"CsPbI3", "MASnI3"}
    assert [j["formula"] for j in body] == sorted(j["formula"] for j in body)


def test_job_inexistente_sigue_dando_404(client):
    assert client.get("/api/jobs/no-existe").status_code == 404


# ── Salud y volumen externo ──────────────────────────────────────────────────

def test_health_detecta_volumen_desmontado(tmp_path):
    """runs/ y calculations/ son symlinks a un volumen externo.

    Desmontado, el poller no ve jobs y todo parece "en calma": health es la
    única señal que distingue eso de "no hay trabajo pendiente".
    """
    app = create_app(config={})
    roto = tmp_path / "volumen-ausente" / "runs" / "relax_basic"
    app.state.poller = StubPoller(roto, [])
    app.state.hub = None

    body = TestClient(app).get("/api/health").json()

    assert body["runs_mounted"] is False
    assert body["ok"] is False
    # Cuelga fuera de la raíz de datos: es el disco externo, no un directorio
    # que el pipeline todavía no haya creado.
    assert body["runs_state"] == "desmontado"
    assert body["n_jobs_tracked"] == 0
    # Señala dónde se rompe la cadena de directorios.
    assert Path(body["nearest_existing_path"]).exists()
    assert body["nearest_existing_path"] == str(tmp_path)


def test_health_ok_con_volumen_montado_y_poller_fresco(client):
    body = client.get("/api/health").json()
    assert body["runs_mounted"] is True
    assert body["ok"] is True
    assert body["n_jobs_tracked"] == len(SNAPSHOTS)
    assert body["last_poll_age_sec"] < 5


def test_health_marca_no_ok_si_el_poller_esta_congelado(tmp_path):
    app = create_app(config={})
    poller = StubPoller(tmp_path, SNAPSHOTS, cfg={"poll_interval_sec": 30})
    poller.last_poll_at = time.time() - 3600  # una hora sin sondear
    app.state.poller = poller
    app.state.hub = None

    body = TestClient(app).get("/api/health").json()
    assert body["runs_mounted"] is True
    assert body["ok"] is False, "un poller congelado debe reportarse como no-ok"


def test_health_no_alarma_en_una_instalacion_sin_estrenar(tmp_path):
    """Recien instalado, runs/ no existe porque no se ha lanzado nada aun.

    El panel decia salud "Atencion" y volumen "DESMONTADO" en rojo sobre una
    instalacion intacta: el mismo patron que el resto de la auditoria, un
    recurso ausente contado como averia.
    """
    paths.set_data_root(tmp_path)
    app = create_app(config={})
    app.state.poller = StubPoller(tmp_path / "runs" / "relax_basic", [])
    app.state.hub = None

    body = TestClient(app).get("/api/health").json()

    assert body["runs_state"] == "sin_estrenar"
    assert body["runs_mounted"] is False
    assert body["ok"] is True, "no hay nada roto, solo nada hecho todavia"


def test_health_sigue_avisando_si_el_volumen_tenia_jobs_y_desaparece(tmp_path):
    """Con jobs ya rastreados, que runs/ se esfume si es una averia."""
    paths.set_data_root(tmp_path)
    app = create_app(config={})
    app.state.poller = StubPoller(tmp_path / "runs" / "relax_basic", SNAPSHOTS)
    app.state.hub = None

    body = TestClient(app).get("/api/health").json()

    assert body["runs_state"] == "desmontado"
    assert body["ok"] is False


# ── Paginación y filtros ─────────────────────────────────────────────────────

def test_jobs_devuelve_sobre_paginado(client):
    body = client.get("/api/jobs").json()
    assert body["total"] == len(SNAPSHOTS)
    assert body["offset"] == 0
    assert len(body["items"]) == len(SNAPSHOTS)


def test_jobs_respeta_limit_y_offset(client):
    primera = client.get("/api/jobs", params={"limit": 2, "offset": 0}).json()
    segunda = client.get("/api/jobs", params={"limit": 2, "offset": 2}).json()

    assert len(primera["items"]) == 2
    assert len(segunda["items"]) == 2
    assert primera["total"] == segunda["total"] == len(SNAPSHOTS)
    assert not {j["job_id"] for j in primera["items"]} & {j["job_id"] for j in segunda["items"]}


def test_jobs_filtra_por_estado(client):
    body = client.get("/api/jobs", params={"status": "converged"}).json()
    assert body["total"] == 2
    assert all(j["status"] == "converged" for j in body["items"])

    varios = client.get("/api/jobs", params={"status": "converged,failed"}).json()
    assert varios["total"] == 3


def test_jobs_busca_por_formula_y_job_id(client):
    por_formula = client.get("/api/jobs", params={"q": "sn"}).json()
    assert {j["formula"] for j in por_formula["items"]} == {"MASnI3", "CsSnI3", "FASnBr3"}

    por_id = client.get("/api/jobs", params={"q": "ccc3"}).json()
    assert por_id["total"] == 1
    assert por_id["items"][0]["formula"] == "FAPbBr3"


def test_jobs_ordena_y_admite_descendente(client):
    asc = client.get("/api/jobs", params={"sort": "elapsed_min"}).json()
    desc = client.get("/api/jobs", params={"sort": "elapsed_min", "desc": True}).json()

    tiempos = [j["elapsed_min"] or 0.0 for j in asc["items"]]
    assert tiempos == sorted(tiempos)
    assert [j["job_id"] for j in desc["items"]] == [j["job_id"] for j in asc["items"]][::-1]


def test_jobs_rechaza_orden_desconocido(client):
    assert client.get("/api/jobs", params={"sort": "energia"}).status_code == 422


# ── Resumen ──────────────────────────────────────────────────────────────────

def test_summary_cuenta_por_estado(client):
    body = client.get("/api/summary").json()
    assert body["total"] == len(SNAPSHOTS)
    assert body["n_converged"] == 2
    assert body["n_failed"] == 1
    assert body["n_running"] == 1
    assert body["n_pending"] == 1
    assert body["convergence_rate"] == pytest.approx(2 / 3, abs=1e-3)


# ── Estados escritos por el pipeline ─────────────────────────────────────────

def test_skipped_duplicate_es_un_estado_valido():
    """37 % de los jobs de Fase 2A usan este estado terminal.

    No estaba en `JobStatusLiteral`, así que `snapshot_job()` lanzaba
    ValidationError, `poll_once` lo tragaba en DEBUG y esos jobs desaparecían
    del monitor entero: API, resumen y reportes de Telegram.
    """
    snap = _snap("fff6", "FA0.5MA0.5PbI3", "skipped_duplicate")
    assert snap.status == "skipped_duplicate"


def test_summary_cuenta_los_skipped_duplicate(tmp_path):
    app = create_app(config={})
    app.state.poller = StubPoller(
        tmp_path,
        [*SNAPSHOTS, _snap("fff6", "FA0.5MA0.5PbI3", "skipped_duplicate")],
    )
    app.state.hub = None

    body = TestClient(app).get("/api/summary").json()
    assert body["n_skipped_duplicate"] == 1
    assert body["total"] == len(SNAPSHOTS) + 1
    # No debe contaminar la tasa de convergencia (convergidos / conv+fallidos).
    assert body["convergence_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_todos_los_status_json_reales_se_parsean():
    """Contra los 550 jobs reales de local_runs/, sin depender del disco externo."""
    from collections import Counter

    from monitor_api.poller import snapshot_job

    raiz = ROOT / "local_runs" / "phase2_force"
    if not raiz.is_dir():
        pytest.skip("local_runs/phase2_force no disponible")

    cfg = {"stall_minutes": 10, "oscillation_window": 10, "oscillation_energy_std_ev": 0.05}
    vistos, fallos = Counter(), []
    for batch in sorted(raiz.glob("batch_*")):
        for d in sorted(batch.iterdir()):
            if d.is_dir() and (d / "status.json").exists():
                try:
                    vistos[snapshot_job(d, cfg).status] += 1
                except Exception as exc:
                    fallos.append((d.name, str(exc)))

    assert not fallos, f"{len(fallos)} jobs reales no se parsean: {fallos[:3]}"
    assert sum(vistos.values()) > 0


# ── WebSocket: EventHub ──────────────────────────────────────────────────────

import asyncio  # noqa: E402
import json  # noqa: E402

from monitor_api import ws as ws_mod  # noqa: E402
from monitor_api.models import WsEvent  # noqa: E402


class StubQueueHolder:
    """Solo aporta el event_queue que consume el hub."""

    def __init__(self):
        self.event_queue: asyncio.Queue = asyncio.Queue()


def _evento(job_id: str, event: str = "CONVERGED") -> WsEvent:
    return WsEvent(job_id=job_id, event=event, timestamp="2026-08-24T12:00:00")


def test_hub_reparte_el_mismo_evento_a_todos_los_clientes():
    """Antes cada cliente lanzaba su propia task sobre la cola compartida."""

    async def run():
        holder = StubQueueHolder()
        hub = ws_mod.EventHub()
        _, c1 = hub.register()
        _, c2 = hub.register()
        hub.start(holder)

        await holder.event_queue.put(_evento("aaa1"))
        m1 = json.loads(await asyncio.wait_for(c1.queue.get(), timeout=2))
        m2 = json.loads(await asyncio.wait_for(c2.queue.get(), timeout=2))

        await hub.stop()
        return m1, m2

    m1, m2 = asyncio.run(run())
    assert m1 == m2
    assert m1["type"] == "event"
    assert m1["job_id"] == "aaa1"
    assert m1["event"] == "CONVERGED"


def test_hub_numera_los_eventos_de_forma_monotona():
    """El `seq` permite al cliente detectar huecos y resincronizar por REST."""

    async def run():
        holder = StubQueueHolder()
        hub = ws_mod.EventHub()
        _, c = hub.register()
        hub.start(holder)

        for i in range(3):
            await holder.event_queue.put(_evento(f"job{i}"))
        seqs = [
            json.loads(await asyncio.wait_for(c.queue.get(), timeout=2))["seq"]
            for _ in range(3)
        ]
        await hub.stop()
        return seqs

    assert asyncio.run(run()) == [1, 2, 3]


def test_hub_contabiliza_los_eventos_descartados():
    """Una cola llena descartaba en silencio; ahora queda registrado."""

    async def run():
        holder = StubQueueHolder()
        hub = ws_mod.EventHub()
        _, c = hub.register()
        hub.start(holder)

        for i in range(ws_mod.CLIENT_QUEUE_MAX + 5):
            await holder.event_queue.put(_evento(f"job{i}"))

        for _ in range(50):          # dar tiempo al pump a vaciar la cola origen
            if holder.event_queue.empty():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

        dropped = c.dropped
        await hub.stop()
        return dropped

    assert asyncio.run(run()) > 0


def test_hub_se_desuscribe_al_cerrar():
    async def run():
        hub = ws_mod.EventHub()
        cid, _ = hub.register()
        assert hub.n_clients == 1
        hub.unregister(cid)
        return hub.n_clients

    assert asyncio.run(run()) == 0


def test_websocket_saluda_y_hace_ping_sin_trafico(monkeypatch):
    """El bucle antiguo prometía un ping cada 15 s en un comentario y nunca lo enviaba."""
    monkeypatch.setattr(ws_mod, "PING_INTERVAL_SEC", 0.1)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        holder = StubQueueHolder()
        hub = ws_mod.EventHub()
        app.state.poller = holder
        app.state.hub = hub
        hub.start(holder)
        yield
        await hub.stop()

    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(ws_mod.router)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as sock:
            assert sock.receive_json() == {"type": "hello", "seq": 0}
            ping = sock.receive_json()
            assert ping["type"] == "ping"


# ── Autenticación ────────────────────────────────────────────────────────────

from monitor_api import security  # noqa: E402

TOKEN = "token-de-prueba-1234567890"


@pytest.fixture
def auth_client(tmp_path):
    """App con auth activa; la auditoría se desvía a tmp_path."""
    security.reset_rate_limit()

    app = create_app(
        config={
            "monitor": {
                "auth": {"token": TOKEN, "session_secret": "secreto-de-prueba"}
            }
        }
    )
    app.state.auth.audit_path = tmp_path / "audit.jsonl"
    app.state.poller = StubPoller(tmp_path, SNAPSHOTS)
    app.state.hub = None
    return TestClient(app)


def test_api_rechaza_sin_sesion(auth_client):
    for ruta in ("/api/health", "/api/jobs", "/api/summary", "/api/jobs/converged"):
        assert auth_client.get(ruta).status_code == 401, ruta


def test_login_correcto_abre_el_resto_de_la_api(auth_client):
    assert auth_client.get("/api/summary").status_code == 401

    r = auth_client.post("/auth/login", json={"token": TOKEN})
    assert r.status_code == 200
    assert r.json() == {"authenticated": True, "auth_enabled": True}

    # La cookie de sesión queda en el cliente.
    assert auth_client.get("/api/summary").status_code == 200
    assert auth_client.get("/api/jobs").json()["total"] == len(SNAPSHOTS)


def test_login_con_token_incorrecto(auth_client):
    r = auth_client.post("/auth/login", json={"token": "no-es"})
    assert r.status_code == 401
    assert auth_client.get("/api/summary").status_code == 401


def test_login_se_bloquea_tras_varios_fallos(auth_client):
    for _ in range(security.MAX_FAILED_ATTEMPTS):
        auth_client.post("/auth/login", json={"token": "no-es"})

    bloqueado = auth_client.post("/auth/login", json={"token": "no-es"})
    assert bloqueado.status_code == 429

    # El bloqueo aplica incluso con el token correcto: es por IP, no por token.
    assert auth_client.post("/auth/login", json={"token": TOKEN}).status_code == 429


def test_logout_cierra_la_sesion(auth_client):
    auth_client.post("/auth/login", json={"token": TOKEN})
    assert auth_client.get("/api/summary").status_code == 200

    auth_client.post("/auth/logout")
    assert auth_client.get("/api/summary").status_code == 401


def test_auth_me_informa_del_estado(auth_client):
    antes = auth_client.get("/auth/me").json()
    assert antes == {"authenticated": False, "auth_enabled": True}

    auth_client.post("/auth/login", json={"token": TOKEN})
    assert auth_client.get("/auth/me").json()["authenticated"] is True


def test_websocket_se_cierra_sin_sesion(auth_client):
    """El navegador no puede mandar cabeceras en el handshake WS.

    Por eso la sesión va en cookie: protege REST y WebSocket con el mismo
    mecanismo.
    """
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with auth_client.websocket_connect("/ws/events"):
            pass


def test_auditoria_registra_los_intentos(auth_client, tmp_path):
    auth_client.post("/auth/login", json={"token": "no-es"})
    auth_client.post("/auth/login", json={"token": TOKEN})

    lineas = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [e["action"] for e in lineas] == ["login_fallido", "login"]
    assert all("ts" in e and "client" in e for e in lineas)


def test_auth_activa_sin_token_es_un_error_explicito():
    """Fallar al arrancar es mejor que quedarse con un monitor imposible de usar."""
    with pytest.raises(ValueError, match="no hay token"):
        security.load_auth_config({"monitor": {"auth": {"enabled": True}}})


def test_sin_token_la_auth_queda_desactivada(monkeypatch):
    monkeypatch.delenv("DFT_MONITOR_TOKEN", raising=False)
    cfg = security.load_auth_config({})
    assert cfg.enabled is False


def test_con_token_la_auth_se_activa_sola(monkeypatch):
    monkeypatch.delenv("DFT_MONITOR_TOKEN", raising=False)
    cfg = security.load_auth_config({"monitor": {"auth": {"token": "abc"}}})
    assert cfg.enabled is True


# ── Historial de métricas ────────────────────────────────────────────────────

from monitor_api.metrics_history import MetricsHistory  # noqa: E402


def test_history_vacio_si_no_hay_muestreador(client):
    """La app tiene que responder aunque el muestreo no haya arrancado."""
    body = client.get("/api/system/history").json()
    assert body == {"samples": [], "interval_sec": 0}


def test_history_devuelve_las_muestras(tmp_path):
    app = create_app(config={})
    app.state.poller = StubPoller(tmp_path, SNAPSHOTS)
    app.state.hub = None

    history = MetricsHistory(maxlen=10)
    history.sample_now()
    history.sample_now()
    app.state.metrics_history = history

    body = TestClient(app).get("/api/system/history").json()
    assert len(body["samples"]) == 2
    assert body["interval_sec"] > 0
    m = body["samples"][0]
    assert {"t", "cpu_percent", "ram_percent", "ram_used_gb", "core_temp_max"} <= set(m)
    assert 0 <= m["cpu_percent"] <= 100


def test_history_recorta_por_ventana_temporal():
    import time as _time
    from dataclasses import replace

    history = MetricsHistory(maxlen=10)
    reciente = history.sample_now()
    # Una muestra de hace una hora no debe salir en la ventana de 10 min.
    history._buf.appendleft(replace(reciente, t=_time.time() - 3600))

    assert len(history.samples()) == 2
    assert len(history.samples(since_sec=600)) == 1


def test_history_es_un_buffer_circular():
    history = MetricsHistory(maxlen=3)
    for _ in range(5):
        history.sample_now()
    assert len(history) == 3


# ── Agente local Ollama ─────────────────────────────────────────────────────

from monitor_api.services.agent import proposals as proposal_svc  # noqa: E402
from monitor_api.services.agent import service as agent_service  # noqa: E402
from monitor_api.services.agent.tools import ReadOnlyToolCatalog  # noqa: E402


def test_agent_health_desactivado_por_default(client):
    body = client.get("/api/agent/health").json()
    assert body["enabled"] is False
    assert body["ok"] is False
    assert body["provider"] == "ollama"


def test_agent_tools_salen_del_openapi_y_solo_son_lectura(client):
    catalog = ReadOnlyToolCatalog(client.app)
    assert catalog.tools
    assert all(t.method == "get" for t in catalog.tools)
    assert not any("kill" in t.path or "retry" in t.path or "start" in t.path for t in catalog.tools)
    assert "get_api_summary" in {t.name for t in catalog.tools}


def test_agent_tool_loop_usa_ollama_fake_y_audita(monkeypatch, tmp_path):
    class FakeOllama:
        payloads: list[dict] = []

        def __init__(self, *args, **kwargs):
            pass

        async def chat(self, payload):
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return {
                    "model": payload["model"],
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "get_api_summary", "arguments": {}}}
                        ],
                    },
                    "done": True,
                }
            return {
                "model": payload["model"],
                "message": {"role": "assistant", "content": "Resumen operativo listo."},
                "done": True,
            }

    monkeypatch.setattr(agent_service, "OllamaClient", FakeOllama)
    app = create_app(
        config={
            "monitor": {
                "agent": {
                    "enabled": True,
                    "provider": "ollama",
                    "model": "dft-agent:14b-q4",
                }
            }
        }
    )
    app.state.auth.audit_path = tmp_path / "audit.jsonl"
    app.state.poller = StubPoller(tmp_path, SNAPSHOTS)
    app.state.hub = None
    client = TestClient(app)

    r = client.post("/api/agent/chat", json={"message": "resume el monitor"})

    assert r.status_code == 200
    body = r.json()
    assert body["message"] == "Resumen operativo listo."
    assert body["tool_results"][0]["name"] == "get_api_summary"
    assert body["tool_results"][0]["ok"] is True
    assert len(FakeOllama.payloads) == 2
    assert FakeOllama.payloads[0]["tools"]

    acciones = [json.loads(x)["action"] for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert "agent_chat" in acciones
    assert "agent_tool_read" in acciones


def test_agent_propuestas_se_aprueban_y_rechazan_con_auditoria(client, tmp_path):
    proposal_svc.reset_proposals()
    client.app.state.auth.audit_path = tmp_path / "audit.jsonl"
    p1 = proposal_svc.create_proposal(title="Reintentar job", command="retry aaa1")
    p2 = proposal_svc.create_proposal(title="No tocar", command="kill bbb2")

    aprobado = client.post(f"/api/agent/proposals/{p1.id}/approve")
    rechazado = client.post(f"/api/agent/proposals/{p2.id}/reject")

    assert aprobado.status_code == 200
    assert aprobado.json()["status"] == "approved"
    assert aprobado.json()["executed"] is False
    assert rechazado.status_code == 200
    assert rechazado.json()["status"] == "rejected"

    acciones = [json.loads(x)["action"] for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert "agent_proposal_approved" in acciones
    assert "agent_proposal_rejected" in acciones


# ── Artefactos de un job ─────────────────────────────────────────────────────

from monitor_api.services import jobs as jobs_svc  # noqa: E402

JOB_REAL = "0a19dca08b812f31"
BATCH_REAL = ROOT / "local_runs" / "phase2_force" / "batch_000"


@pytest.fixture
def batch_client():
    """Cliente apuntando al batch real de local_runs/."""
    if not (BATCH_REAL / JOB_REAL).is_dir():
        pytest.skip("local_runs/phase2_force/batch_000 no disponible")
    app = create_app(config={})
    app.state.poller = StubPoller(BATCH_REAL, SNAPSHOTS)
    app.state.hub = None
    return TestClient(app)


@pytest.mark.parametrize(
    "job_id",
    ["../../etc/passwd", "..", ".", "a/b", f"{JOB_REAL}/../..", "", "x" * 200, "-arranca-con-guion"],
)
def test_resolve_job_dir_bloquea_ids_hostiles(tmp_path, job_id):
    """El job_id llega por la URL: `runs_dir / job_id` con '..' saldría del árbol."""
    with pytest.raises(jobs_svc.UnsafeJobIdError):
        jobs_svc.resolve_job_dir(tmp_path, job_id)


def test_resolve_job_dir_devuelve_none_si_no_existe(tmp_path):
    assert jobs_svc.resolve_job_dir(tmp_path, "no-existe") is None


def test_log_rechaza_job_id_hostil(batch_client):
    # Los clientes HTTP colapsan '..' en la URL antes de enviarla, así que un
    # traversal literal nunca llega al handler; lo que sí llega son ids raros.
    assert batch_client.get("/api/jobs/abc..def/log").status_code == 404
    assert batch_client.get("/api/jobs/.hidden/log").status_code in (400, 404)


def test_la_api_no_queda_enmascarada_por_el_spa(batch_client):
    """Un endpoint mal escrito debe dar 404 JSON, no el shell HTML con 200."""
    r = batch_client.get("/api/no-existe")
    assert r.status_code == 404
    assert "text/html" not in r.headers.get("content-type", "")

    # Las rutas del frontend sí caen en el shell.
    spa = batch_client.get("/jobs")
    assert spa.status_code == 200
    assert "text/html" in spa.headers.get("content-type", "")


def test_log_devuelve_la_cola_y_las_etiquetas(batch_client):
    body = batch_client.get(f"/api/jobs/{JOB_REAL}/log", params={"tail": 5}).json()
    assert body["label"] == "pbe"
    assert body["available"] == ["pbe"]
    assert len(body["lines"]) == 5
    assert body["total_lines"] > 5


def test_log_limita_el_tail(batch_client):
    assert batch_client.get(f"/api/jobs/{JOB_REAL}/log", params={"tail": 99999}).status_code == 422


def test_traces_parsea_las_iteraciones_scf(batch_client):
    body = batch_client.get(f"/api/jobs/{JOB_REAL}/traces").json()

    assert len(body["labels"]) == 1
    pbe = body["labels"][0]
    assert pbe["label"] == "pbe"
    assert pbe["n_iters"] > 0
    assert pbe["rate_s_per_iter"] > 0

    pt = pbe["points"][0]
    assert {"iter", "clock", "energy"} <= set(pt)
    assert [p["iter"] for p in pbe["points"]] == sorted(p["iter"] for p in pbe["points"])

    # La energía converge: las últimas iteraciones se estabilizan. (No vale
    # exigir monotonía — el SCF arranca de una densidad de prueba cuya energía
    # puede quedar por debajo de la convergida.)
    energias = [p["energy"] for p in pbe["points"]]
    spread_final = max(energias[-3:]) - min(energias[-3:])
    spread_inicial = max(energias[:3]) - min(energias[:3])
    assert spread_final < spread_inicial


def test_traces_incluye_los_frames_etiquetados(batch_client):
    frames = batch_client.get(f"/api/jobs/{JOB_REAL}/traces").json()["frames"]
    assert len(frames) == 4
    assert [f["config_index"] for f in frames] == [0, 1, 2, 3]
    assert all(f["status"] == "ok" for f in frames)
    assert all(f["energy_ev"] < 0 for f in frames)
    assert all(f["n_atoms"] == 40 for f in frames)


def test_metadata_expone_la_ficha_del_candidato(batch_client):
    body = batch_client.get(f"/api/jobs/{JOB_REAL}/metadata").json()
    md = body["metadata"]
    assert md["formula"] == "FAPb0.12Sn0.88Cl3"
    assert md["n_atoms"] == 40
    assert md["generation_mode"] == "B_mixed"
    assert body["status"]["status"] == "converged"
    assert any(a.endswith("structure.cif") for a in body["artifacts"])


def test_artefactos_de_job_inexistente_dan_404(batch_client):
    for ruta in ("log", "traces", "metadata"):
        assert batch_client.get(f"/api/jobs/nada/{ruta}").status_code == 404


def test_scf_rate_funciona_con_pocas_iteraciones():
    """Regresión: el emparejamiento antiguo se desalineaba con <6 puntos.

    `zip(points[-6:-1], points[-5:])` comparaba cada punto consigo mismo, todos
    los deltas salían 0 y el ritmo quedaba en None justo en los jobs cortos.
    """
    puntos = [
        {"iter": 1, "clock": "10:00:00"},
        {"iter": 2, "clock": "10:00:30"},
        {"iter": 3, "clock": "10:01:00"},
    ]
    assert jobs_svc._scf_rate_s(puntos) == pytest.approx(30.0)


def test_scf_rate_maneja_el_cruce_de_medianoche():
    puntos = [
        {"iter": 1, "clock": "23:59:00"},
        {"iter": 2, "clock": "23:59:30"},
        {"iter": 3, "clock": "00:00:10"},  # +40 s cruzando medianoche
    ]
    rate = jobs_svc._scf_rate_s(puntos)
    assert rate is not None and 30 <= rate <= 40


def test_scf_rate_sin_datos_suficientes():
    assert jobs_svc._scf_rate_s([]) is None
    assert jobs_svc._scf_rate_s([{"iter": 1, "clock": "10:00:00"}]) is None


# ── Candidatos ───────────────────────────────────────────────────────────────

from monitor_api.services import candidates as cand_svc  # noqa: E402


def test_candidatos_se_reconstruyen_desde_los_jobs(batch_client):
    """data/processed/ está en .gitignore y vive en el volumen externo.

    Sin fallback, la vista quedaría vacía aunque los metadata.json de los jobs
    lleven la selection_row completa. Se pide sin filtrar porque el listado por
    defecto ya no sale de un único `runs_dir`.
    """
    body = batch_client.get(
        "/api/candidates", params={"limit": 5, "verified_only": False}).json()

    assert body["total"] == 50
    assert "batch_000" in body["source"]
    assert len(body["items"]) == 5

    c = body["items"][0]
    assert c["formula"] and c["score"] is not None
    assert 0 < c["tolerance_t"] < 2


def test_por_defecto_solo_los_verificados(batch_client):
    """La vista mostraba los candidatos de un único lote, fijado al arrancar.

    Ahora recorre todos y deja solo lo que cumple las tres condiciones: pasó
    los tiers del cribado, terminó en DFT, y su bandgap cae cerca de la ventana
    fotovoltaica.
    """
    body = batch_client.get("/api/candidates", params={"limit": 200}).json()
    if body["total"] == 0:
        pytest.skip("no hay lotes con jobs convergidos en este entorno")

    for c in body["items"]:
        assert c["dft_status"] == "converged", c["formula"]
        if c.get("pv_score") is not None:
            assert c["pv_score"] >= 0.5

    # Ordenado por score fotovoltaico descendente.
    scores = [c["pv_score"] for c in body["items"] if c.get("pv_score") is not None]
    assert scores == sorted(scores, reverse=True)


def test_el_umbral_pv_recorta(batch_client):
    laxo = batch_client.get("/api/candidates", params={"pv_min": 0.0, "limit": 500}).json()
    estricto = batch_client.get("/api/candidates", params={"pv_min": 0.95, "limit": 500}).json()
    if laxo["total"] == 0:
        pytest.skip("sin candidatos verificados")
    assert estricto["total"] <= laxo["total"]
    assert all(x["has_dft"] for x in laxo["items"])


def test_candidatos_ordenan_por_score_descendente(batch_client):
    items = batch_client.get("/api/candidates", params={"limit": 50}).json()["items"]
    scores = [c["score"] for c in items if c["score"] is not None]
    assert scores == sorted(scores, reverse=True)


def test_candidatos_exponen_facetas_y_cotas_de_filtro(batch_client):
    body = batch_client.get("/api/candidates").json()

    # Facetas reales del batch, para poblar los desplegables de la GUI.
    assert set(body["facets"]["generation_mode"]) <= {"pure", "A_mixed", "B_mixed", "X_mixed"}
    assert body["facets"]["dominant_halide"]

    # Cotas de config/generator.yaml, para dibujar las zonas de aceptación.
    assert body["filters"]["goldschmidt"] == {"min": 0.80, "max": 1.10}
    assert body["filters"]["octahedral"] == {"min": 0.40, "max": 0.90}


def test_candidatos_filtran_por_haluro(batch_client):
    """El filtro debe recortar de verdad, no solo pasar la petición.

    Antes se comprobaba `total < 50`, que era el tamaño de un único lote: al
    listar todos, ese número dejó de significar nada.
    """
    todos = batch_client.get("/api/candidates", params={"limit": 5000}).json()
    body = batch_client.get(
        "/api/candidates", params={"halide": "I", "limit": 5000}).json()

    assert 0 < body["total"] <= todos["total"]
    assert all(c["dominant_halide"] == "I" for c in body["items"])

    otros = {c["dominant_halide"] for c in todos["items"]} - {"I", None}
    if otros:
        assert body["total"] < todos["total"], "el filtro no recortó nada"


def test_candidatos_sin_origen_no_revientan(tmp_path):
    filas, origen = cand_svc.load_candidates(tmp_path / "no-existe")
    assert filas == []
    assert origen == "sin datos"


# ── Surrogate ML ─────────────────────────────────────────────────────────────

from monitor_api.services import ml as ml_svc  # noqa: E402

_MODELO_OK = (ROOT / "models" / "surrogate_bandgap.pkl").is_file()


def test_models_lista_las_metricas(client):
    body = client.get("/api/models").json()
    nombres = {m["name"] for m in body["models"]}
    assert "surrogate_bandgap" in nombres
    assert body["surrogate_status"] in ("ok", "error")
    # Si falla la carga, el motivo tiene que ser accionable.
    if body["surrogate_status"] == "error":
        assert "scikit-learn" in body["surrogate_error"]


@pytest.mark.skipif(not _MODELO_OK, reason="models/surrogate_bandgap.pkl no disponible")
def test_predict_devuelve_bandgap_con_incertidumbre(client):
    ml_svc.reset_cache()
    r = client.post("/api/ml/predict", json={"A": "Cs", "B": "Pb", "X": "I"})
    if r.status_code == 503:
        pytest.skip(f"surrogate no cargable: {r.json()['detail'][:80]}")

    body = r.json()
    assert body["material"] == "CsPbI3"
    assert 0 < body["bandgap_pred"] < 5
    assert body["bandgap_uncertainty"] >= 0
    # CsPbI3 experimental ≈ 1.73 eV; el modelo debe caer razonablemente cerca.
    assert abs(body["bandgap_pred"] - 1.73) < 0.6


@pytest.mark.skipif(not _MODELO_OK, reason="models/surrogate_bandgap.pkl no disponible")
def test_top8_compara_ml_dft_y_experimento(client):
    ml_svc.reset_cache()
    body = client.get("/api/ml/top8").json()
    assert len(body["items"]) == 8

    fila = next(i for i in body["items"] if i["material"] == "CsPbI3")
    assert fila["Eg_exp_eV"] == 1.73
    assert fila["Eg_dft_eV"] is not None
    if "error" not in fila:
        assert fila["Eg_ml_eV"] is not None


def test_surrogate_indisponible_da_503(client, monkeypatch):
    """Un pickle incompatible es un fallo operativo real, no un 500 opaco."""
    ml_svc.reset_cache()
    monkeypatch.setattr(
        ml_svc, "_load", lambda: (_ for _ in ()).throw(ml_svc.SurrogateUnavailableError("pickle roto"))
    )
    r = client.post("/api/ml/predict", json={"A": "Cs", "B": "Pb", "X": "I"})
    assert r.status_code == 503
    assert "pickle roto" in r.json()["detail"]
    ml_svc.reset_cache()


# ── Estructuras, reportes y figuras ──────────────────────────────────────────

from monitor_api.services import files as files_svc  # noqa: E402


@pytest.mark.parametrize(
    "relative",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "../pyproject.toml",
        "top8/../../pyproject.toml",
        "a/../../b",
        "",
    ],
)
def test_safe_join_bloquea_traversal(relative):
    """Estos endpoints sirven archivos elegidos por el cliente."""
    with pytest.raises(files_svc.UnsafePathError):
        files_svc.safe_join(files_svc.structures_dir(), relative)


def test_safe_join_aplica_lista_blanca_de_extensiones():
    with pytest.raises(files_svc.UnsafePathError):
        files_svc.safe_join(paths.data_root(), "pyproject.toml", files_svc.EXT_REPORTE)


def test_safe_join_acepta_una_ruta_legitima():
    raiz = files_svc.structures_dir()
    p = files_svc.safe_join(raiz, "top8/CsPbI3.cif", files_svc.EXT_ESTRUCTURA)
    assert p.name == "CsPbI3.cif"
    assert raiz.resolve() in p.parents


def test_structures_lista_las_tres_procedencias(batch_client):
    items = batch_client.get("/api/structures").json()["items"]
    grupos = {i["group"] for i in items}
    assert {"fases", "top8", "jobs"} <= grupos
    assert any(i["id"] == "repo:alpha_cubic.json" for i in items)
    assert any(i["group"] == "jobs" for i in items)


def test_todo_lo_listado_se_puede_abrir(batch_client):
    """El id de cada estructura tiene que resolver.

    Antes el grupo «jobs» era siempre el `runs_dir` configurado y sus ids iban
    con prefijo `job:`, que se resuelve contra ese directorio. Ahora la sección
    sigue al lote que se está calculando, que puede ser otro: si el prefijo no
    acompañara, la lista enseñaría entradas que al pulsarlas dan 404.
    """
    items = batch_client.get("/api/structures").json()["items"]
    muestra = [items[0], items[len(items) // 2], items[-1]]
    for grupo in ("jobs", "fases", "top8"):
        primero = next((i for i in items if i["group"] == grupo), None)
        if primero:
            muestra.append(primero)

    for item in muestra:
        r = batch_client.get("/api/structures/content", params={"id": item["id"]})
        assert r.status_code == 200, f'{item["id"]} ({item["group"]}) -> {r.status_code}'
        assert "_cell_length_a" in r.json()["content"]


def test_structures_expone_solo_las_seleccionadas_mas_recientes(tmp_path):
    """El visor muestra en recientes solo estructuras que pasaron la selección ML."""
    from monitor_api import paths

    paths.set_data_root(tmp_path)
    batch = tmp_path / "phase2_force" / "batch_042"
    job = batch / "abc123def456"
    job.mkdir(parents=True)
    (job / "metadata.json").write_text(json.dumps({
        "formula": "MAPbI3",
        "A_site_species": ["MA"],
        "B_site_species": ["Pb"],
        "X_site_species": ["I"],
        "molecular_A_placeholder": True,
        "selection_row": {
            "source_type": "cascade",
            "source_file": "data/batches/batch_042/selected_for_dft.csv",
        },
    }))
    (job / "structure.cif").write_text("data_recent\n_cell_length_a 6.0\n")

    raw_job = batch / "raw123def456"
    raw_job.mkdir()
    (raw_job / "metadata.json").write_text(json.dumps({
        "formula": "CsPbI3",
    }))
    (raw_job / "structure.cif").write_text("data_raw\n_cell_length_a 6.1\n")

    app = create_app(config={})
    app.state.poller = StubPoller(batch, [])
    app.state.hub = None
    test_client = TestClient(app)

    items = test_client.get("/api/structures").json()["items"]
    # Sin selección ML no se lista: el visor solo enseña lo seleccionado.
    assert not any("raw123def456" in i["id"] for i in items)
    # Con un único lote, el activo es ese, así que entra como lote en curso.
    reciente = next(i for i in items if "abc123def456" in i["id"])
    assert reciente["group"] in ("jobs", "recientes")
    assert reciente["name"] == "MAPbI3"
    assert "batch_042" in reciente["detail"]

    body = test_client.get("/api/structures/content", params={"id": reciente["id"]}).json()
    assert body["name"] == "MAPbI3"
    assert "_cell_length_a" in body["content"]
    assert body["metadata"]["molecular_A_placeholder"] is True
    assert "placeholder" in body["metadata"]["organic_A_warning"]


def test_structure_content_convierte_json_de_ase_a_cif(batch_client):
    """El visor necesita CIF; structures/*.json es el formato de base de ASE."""
    body = batch_client.get("/api/structures/content", params={"id": "repo:alpha_cubic.json"}).json()
    assert body["format"] == "cif"
    assert body["content"].startswith("data_")
    assert "_cell_length_a" in body["content"]
    assert "6.2965" in body["content"]


def test_structure_content_sirve_el_cif_de_un_job(batch_client):
    body = batch_client.get("/api/structures/content", params={"id": f"job:{JOB_REAL}"}).json()
    assert "_cell_length_a" in body["content"]
    assert "_atom_site" in body["content"]


def test_structure_content_rechaza_identificadores_hostiles(batch_client):
    for ident in ["repo:../pyproject.toml", "job:../../etc", "otra-cosa"]:
        r = batch_client.get("/api/structures/content", params={"id": ident})
        assert r.status_code in (400, 404), ident


def test_reports_lista_documentos_y_galerias(client):
    body = client.get("/api/reports").json()
    assert any(d["path"].endswith(".md") for d in body["documents"])
    assert body["galleries"]

    g = body["galleries"][0]
    assert {"n_declared", "n_present", "figures"} <= set(g)
    # Los PNG/PDF están en .gitignore: n_present puede ser 0 y es lo normal.
    assert g["n_present"] <= g["n_declared"]


def test_report_document_devuelve_el_markdown(client):
    docs = client.get("/api/reports").json()["documents"]
    if not docs:
        pytest.skip("sin reportes Markdown")
    body = client.get("/api/reports/document", params={"path": docs[0]["path"]}).json()
    assert body["content"]
    assert body["name"].endswith(".md")


def test_report_document_rechaza_rutas_fuera_del_arbol(client):
    for ruta in ["pyproject.toml", "../etc/passwd", "src/monitor_api/security.py"]:
        assert client.get("/api/reports/document", params={"path": ruta}).status_code in (400, 404)


def test_figura_ausente_explica_como_regenerarla(client):
    """0 de 52 figuras están en disco tras un clon; el mensaje debe ser útil."""
    r = client.get("/api/reports/figure", params={"path": "imagenes/dos_pdos_from_gpw.png"})
    assert r.status_code == 404
    assert "generate_visualizations" in r.json()["detail"]


# ── Control: kill, retry, batches ────────────────────────────────────────────

import os  # noqa: E402
import shutil  # noqa: E402

from monitor_api.services import control as ctl  # noqa: E402


def _job_falso(tmp_path, estado: str, **extra):
    d = tmp_path / "aaa1"
    d.mkdir(exist_ok=True)
    (d / "status.json").write_text(
        json.dumps({"status": estado, "candidate_id": "aaa1", "formula": "CsPbI3", **extra})
    )
    return d


# ── Verificación del PID ─────────────────────────────────────────────────────

def test_pid_verificado_rechaza_un_pid_ajeno(tmp_path):
    """status.json guarda el PID de la última ejecución, quizá de hace meses.

    En Linux los PID se reciclan; pasar ese número a _kill_job_processes()
    mataría el GRUPO de procesos de algo sin relación con el job.
    """
    job = _job_falso(tmp_path, "running")
    # Este proceso existe pero su cwd no cuelga del directorio del job.
    assert ctl._pid_verificado(job, os.getpid()) is None


def test_pid_verificado_rechaza_pids_imposibles(tmp_path):
    job = _job_falso(tmp_path, "running")
    for pid in (None, 0, -1, "123", 4_000_000):
        assert ctl._pid_verificado(job, pid) is None


def test_pid_verificado_acepta_un_proceso_del_job():
    # El cwd de este proceso es la raíz del repo, así que sirve como "job_dir".
    assert ctl._pid_verificado(Path.cwd(), os.getpid()) == os.getpid()


# ── kill ─────────────────────────────────────────────────────────────────────

def test_kill_rechaza_un_job_ya_terminado(tmp_path, monkeypatch):
    job = _job_falso(tmp_path, "converged")
    llamado = []
    monkeypatch.setattr("monitor_api.poller._kill_job_processes", lambda *a: llamado.append(a))

    with pytest.raises(ctl.ControlError, match="converged"):
        asyncio.run(ctl.kill_job(job))
    assert not llamado, "no debe tocar procesos de un job terminado"


def test_kill_no_propaga_un_pid_sin_verificar(tmp_path, monkeypatch):
    """Regresión de seguridad: el PID viejo no debe llegar al matador."""
    job = _job_falso(tmp_path, "running", pid=os.getpid())
    recibidos = []

    def falso_kill(job_dir, root_pid=None):
        recibidos.append(root_pid)
        return [111, 222]

    monkeypatch.setattr("monitor_api.poller._kill_job_processes", falso_kill)
    resultado = asyncio.run(ctl.kill_job(job))

    assert recibidos == [None], "un PID no verificado debe pasar como None"
    assert resultado["killed_pids"] == [111, 222]
    assert json.loads((job / "status.json").read_text())["status"] == "stopped"


def test_kill_marca_el_job_y_borra_el_pid(tmp_path, monkeypatch):
    job = _job_falso(tmp_path, "running", pid=999999)
    monkeypatch.setattr("monitor_api.poller._kill_job_processes", lambda *a, **k: [])

    asyncio.run(ctl.kill_job(job))
    st = json.loads((job / "status.json").read_text())
    assert st["status"] == "stopped"
    assert "pid" not in st
    assert st["stopped_by"] == "monitor-api"


# ── retry ────────────────────────────────────────────────────────────────────

def test_retry_devuelve_el_job_a_la_cola(tmp_path):
    job = _job_falso(
        tmp_path, "failed", pid=123, started_at="2026-06-11T10:00:00Z", error="boom"
    )
    r = ctl.retry_job(job)

    st = json.loads((job / "status.json").read_text())
    assert st["status"] == "pending"
    assert st["requeue_count"] == 1
    assert not {"pid", "started_at", "error"} & set(st)
    assert r["status"] == "pending"

    # Un job ya en cola no se puede reintentar otra vez.
    with pytest.raises(ctl.ControlError, match="pending"):
        ctl.retry_job(job)

    # Si vuelve a fallar, el contador acumula.
    st["status"] = "failed"
    (job / "status.json").write_text(json.dumps(st))
    ctl.retry_job(job)
    assert json.loads((job / "status.json").read_text())["requeue_count"] == 2


def test_retry_rechaza_un_job_convergido(tmp_path):
    job = _job_falso(tmp_path, "converged")
    with pytest.raises(ctl.ControlError, match="converged"):
        ctl.retry_job(job)
    assert json.loads((job / "status.json").read_text())["status"] == "converged"


# ── Endpoints ────────────────────────────────────────────────────────────────

@pytest.fixture
def control_client(tmp_path):
    """Cliente sobre una copia del batch real, para poder mutar status.json."""
    if not (BATCH_REAL / JOB_REAL).is_dir():
        pytest.skip("batch real no disponible")
    copia = tmp_path / "batch_000"
    shutil.copytree(BATCH_REAL, copia)

    app = create_app(config={})
    app.state.poller = StubPoller(copia, SNAPSHOTS)
    app.state.hub = None
    app.state.auth.audit_path = tmp_path / "audit.jsonl"
    return TestClient(app), copia


def test_endpoint_kill_devuelve_409_si_el_estado_no_lo_permite(control_client):
    client, _ = control_client
    r = client.post(f"/api/jobs/{JOB_REAL}/kill")   # ese job está converged
    assert r.status_code == 409
    assert "converged" in r.json()["detail"]


def test_endpoint_retry_requeue_y_audita(control_client, tmp_path):
    client, copia = control_client
    fallido = next(
        d.name
        for d in copia.iterdir()
        if d.is_dir() and json.loads((d / "status.json").read_text()).get("status") == "failed"
    )

    r = client.post(f"/api/jobs/{fallido}/retry")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert json.loads((copia / fallido / "status.json").read_text())["status"] == "pending"

    auditoria = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert auditoria[-1]["action"] == "retry"
    assert auditoria[-1]["job_id"] == fallido


def test_las_acciones_rechazadas_tambien_se_auditan(control_client, tmp_path):
    client, _ = control_client
    client.post(f"/api/jobs/{JOB_REAL}/kill")
    acciones = [
        json.loads(x)["action"] for x in (tmp_path / "audit.jsonl").read_text().splitlines()
    ]
    assert "kill_rechazado" in acciones


def test_control_sobre_job_inexistente_da_404(control_client):
    client, _ = control_client
    assert client.post("/api/jobs/nada/kill").status_code == 404
    assert client.post("/api/jobs/nada/retry").status_code == 404


def test_batches_cuenta_por_estado(control_client):
    client, _ = control_client
    body = client.get("/api/batches").json()
    assert body["items"], "debería encontrar batch_000"

    b = next(x for x in body["items"] if x["name"] == "batch_000")
    assert b["batch_id"] == 0
    assert b["total"] == 50
    assert b["counts"]["converged"] == 40
    assert b["is_current"] is True


def test_start_batch_inexistente_da_404(control_client):
    client, _ = control_client
    assert client.post("/api/batches/999/start").status_code == 404


# ── Lanzador ─────────────────────────────────────────────────────────────────

from monitor_api import launcher, paths  # noqa: E402


@pytest.mark.parametrize(
    "host,esperado",
    [("127.0.0.1", True), ("localhost", True), ("::1", True),
     ("0.0.0.0", False), ("192.168.1.50", False)],
)
def test_es_local(host, esperado):
    assert launcher._es_local(host) is esperado


def test_elige_el_primer_runs_dir_que_existe(monkeypatch, tmp_path):
    """runs/ es un symlink a un volumen externo que puede no estar montado."""
    paths.set_data_root(tmp_path)
    monkeypatch.setattr(
        launcher, "RUNS_CANDIDATOS", ("no/existe", "tampoco", "si/existe")
    )
    (tmp_path / "si" / "existe").mkdir(parents=True)

    assert launcher._elegir_runs_dir() == "si/existe"


def test_si_ninguno_existe_devuelve_el_primero(monkeypatch, tmp_path):
    paths.set_data_root(tmp_path)
    monkeypatch.setattr(launcher, "RUNS_CANDIDATOS", ("a", "b"))
    assert launcher._elegir_runs_dir() == "a"


@pytest.fixture
def launcher_aislado(monkeypatch, tmp_path):
    """Redirige datos y configuración a un temporal.

    Se manipula el módulo `paths` —el mecanismo real— en vez de parchear
    constantes del lanzador. Sin esto, los tests reescribirían el
    configs/monitor.yaml del usuario.
    """
    paths.set_data_root(tmp_path)
    monkeypatch.setenv("DFT_MONITOR_CONFIG_DIR", str(tmp_path / "configs"))
    return tmp_path


def test_config_se_crea_desde_el_ejemplo(launcher_aislado):
    ruta, token = launcher.preparar_config("127.0.0.1")

    assert ruta.exists()
    cfg = yaml.safe_load(ruta.read_text())
    assert "monitor" in cfg and "telegram" in cfg
    # En localhost no se genera token: una pantalla de login ahí no protege nada.
    assert token is None
    assert cfg["monitor"]["auth"]["token"] == ""


def test_config_genera_token_si_se_expone_en_la_red(launcher_aislado):
    _, token = launcher.preparar_config("0.0.0.0")

    assert token and len(token) >= 32
    cfg = yaml.safe_load(launcher.config_file().read_text())
    assert cfg["monitor"]["auth"]["token"] == token


def test_config_apunta_a_un_runs_dir_que_existe(launcher_aislado, monkeypatch):
    monkeypatch.setattr(
        launcher, "RUNS_CANDIDATOS", ("runs/relax_basic", "local_runs/lote")
    )
    (launcher_aislado / "local_runs" / "lote").mkdir(parents=True)

    launcher.preparar_config("127.0.0.1")
    cfg = yaml.safe_load(launcher.config_file().read_text())
    assert cfg["monitor"]["runs_dir"] == "local_runs/lote"


def test_config_existente_no_se_pisa(launcher_aislado):
    destino = launcher.config_file()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("monitor: {runs_dir: mio}\n")
    ruta, token = launcher.preparar_config("0.0.0.0")

    assert token is None
    assert yaml.safe_load(ruta.read_text())["monitor"]["runs_dir"] == "mio"


def test_launcher_ready_payload_para_flutter(launcher_aislado):
    args = type("Args", (), {"host": "127.0.0.1", "port": 54321})()
    payload = launcher._ready_payload(args, "http://127.0.0.1:54321")

    assert payload["event"] == "ready"
    assert payload["base_url"] == "http://127.0.0.1:54321"
    assert payload["pid"]
    assert payload["data_root"] == str(launcher_aislado.resolve())
    assert payload["config_dir"]
    assert payload["frozen"] is False


def test_el_ejemplo_versionado_sigue_teniendo_los_anclajes():
    """preparar_config() sustituye texto literal del ejemplo.

    Si alguien reformatea monitor.example.yaml, la sustitución falla en
    silencio y el config generado apunta al volumen desmontado.
    """
    texto = paths.example_config().read_text()
    assert "  runs_dir: runs/relax_basic" in texto
    assert '    token: ""' in texto


# ── Cribado HTS ──────────────────────────────────────────────────────────────

from monitor_api.services import screening as scr_svc  # noqa: E402

_HAY_CONFIG = (ROOT / "config" / "generator.yaml").is_file()


@pytest.fixture(autouse=True)
def _cribado_limpio():
    scr_svc.reset_runs()
    yield
    scr_svc.reset_runs()


def test_tiers_declaran_su_disponibilidad_real(client):
    """La interfaz esconde lo que no se puede ejecutar; necesita saberlo."""
    body = client.get("/api/screening/config").json()
    tiers = {t["tier"]: t for t in body["tiers"]}

    assert set(tiers) == {0, 1, 2}
    assert tiers[0]["available"] is True          # solo descriptores, siempre
    for t in tiers.values():
        # Un tier no disponible tiene que explicar por qué.
        assert t["available"] or t["reason"]


def test_config_expone_las_cotas_que_aplica_la_cascada(client):
    if not _HAY_CONFIG:
        pytest.skip("config/generator.yaml no disponible")

    gates = client.get("/api/screening/config").json()["gates"]
    assert gates["goldschmidt"] == {"min": 0.80, "max": 1.10}
    assert gates["octahedral"] == {"min": 0.40, "max": 0.90}
    assert gates["pv_window"] == [1.1, 1.8]
    assert set(gates["chemical_space"]) == {"A", "B", "X"}
    assert "Pb" in gates["chemical_space"]["B"]


def test_run_rechaza_tamanos_absurdos(client):
    for n in (0, -5, 99999):
        r = client.post("/api/screening/run", json={"batch_id": 0, "n_candidates": n})
        assert r.status_code == 422, n
    assert client.post(
        "/api/screening/run",
        json={"random_seed": 12, "n_batches": 0, "n_candidates": 10},
    ).status_code == 422
    assert client.post(
        "/api/screening/run",
        json={"random_seed": 12, "n_batches": 10, "n_candidates": 501},
    ).status_code == 422


def test_run_inexistente_da_404(client):
    assert client.get("/api/screening/runs/nada").status_code == 404


def test_start_dft_inexistente_da_404(client):
    r = client.post("/api/screening/runs/nada/start-dft", json={"start_runner": False})
    assert r.status_code == 404


def test_screening_start_dft_endpoint_audita(client, tmp_path, monkeypatch):
    client.app.state.auth.audit_path = tmp_path / "audit.jsonl"

    def fake_start_dft(poller, run_id: str, *, start_runner: bool = True) -> dict:
        assert run_id == "abc"
        assert start_runner is False
        return {
            "run_id": run_id,
            "batch_id": 7,
            "batch_path": str(tmp_path / "batch_007"),
            "n_selected": 1,
            "n_prepared": 1,
            "n_existing_or_skipped": 0,
            "runner_launched": False,
            "runner_kind": "relax",
            "runner_error": None,
        }

    monkeypatch.setattr(scr_svc, "start_dft_for_run", fake_start_dft)

    r = client.post("/api/screening/runs/abc/start-dft", json={"start_runner": False})
    assert r.status_code == 200
    assert r.json()["batch_id"] == 7

    acciones = [json.loads(x)["action"] for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert "screening_start_dft" in acciones


@pytest.mark.skipif(not _HAY_CONFIG, reason="config/generator.yaml no disponible")
def test_cascada_completa_de_punta_a_punta(client):
    """Ejecuta la cascada de verdad sobre un lote pequeño."""
    r = client.post("/api/screening/run", json={"batch_id": 97, "n_candidates": 12})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert "items" not in r.json(), "el POST no debe devolver el ranking entero"

    for _ in range(120):
        detalle = client.get(f"/api/screening/runs/{run_id}").json()
        if detalle["status"] in ("done", "error"):
            break
        time.sleep(1)

    assert detalle["status"] == "done", detalle.get("error")
    assert detalle["tiers"], "el embudo no puede venir vacío"

    tier0 = next(t for t in detalle["tiers"] if t["tier"] == 0)
    assert tier0["kind"] == "gate"
    assert tier0["n_in"] >= tier0["n_out"], "el Tier 0 solo puede reducir"

    if detalle["items"]:
        primero = detalle["items"][0]
        assert primero["formula"]
        assert primero["total_score"] is not None
        # El ranking viene ordenado de mayor a menor.
        scores = [i["total_score"] for i in detalle["items"] if i["total_score"] is not None]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not _HAY_CONFIG, reason="config/generator.yaml no disponible")
def test_cascada_usa_semilla_y_numero_de_lotes(client):
    r = client.post(
        "/api/screening/run",
        json={"random_seed": 1234, "n_batches": 2, "n_candidates": 6},
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    for _ in range(120):
        detalle = client.get(f"/api/screening/runs/{run_id}").json()
        if detalle["status"] in ("done", "error"):
            break
        time.sleep(1)

    assert detalle["status"] == "done", detalle.get("error")
    assert detalle["random_seed"] == 1234
    assert detalle["n_batches"] == 2
    assert detalle["n_candidates_per_batch"] == 6
    assert detalle["n_requested"] == 12
    assert detalle["lot_ids"] == [0, 1]


@pytest.mark.skipif(not _HAY_CONFIG, reason="config/generator.yaml no disponible")
def _cribar(client, batch_id: int, n: int = 12) -> dict:
    run_id = client.post(
        "/api/screening/run", json={"batch_id": batch_id, "n_candidates": n}
    ).json()["run_id"]
    for _ in range(180):
        detalle = client.get(f"/api/screening/runs/{run_id}").json()
        if detalle["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert detalle["status"] == "done", detalle.get("error")
    return detalle


@pytest.mark.skipif(not _HAY_CONFIG, reason="config/generator.yaml no disponible")
def test_start_dft_for_run_prepara_jobs_sin_lanzar_runner(tmp_path):
    from buho.generator.heuristic_generator import HeuristicGenerator

    batch_root = tmp_path / "batches"
    batch_root.mkdir()
    candidato = HeuristicGenerator(ROOT / "config" / "generator.yaml").generate_batch(
        42,
        batch_size=1,
    )[0]
    run = scr_svc.ScreeningRun(
        run_id="run42",
        batch_id=42,
        n_requested=1,
        use_mlff=False,
        status="done",
        n_selected=1,
        selected_candidate_ids=[candidato.candidate_id],
        selected_candidates=[candidato.to_dict()],
    )
    scr_svc._runs[run.run_id] = run

    poller = StubPoller(
        batch_root / "batch_000",
        [],
        cfg={
            "phase2_runs_dir": str(batch_root),
            "runner_cores": 1,
            "runner_kind": "relax",
        },
    )

    result = scr_svc.start_dft_for_run(poller, run.run_id, start_runner=False)

    job_dir = batch_root / "batch_042" / candidato.candidate_id
    assert result["n_prepared"] == 1
    assert result["runner_launched"] is False
    assert job_dir.is_dir()
    assert (job_dir / "structure.cif").is_file()
    assert json.loads((job_dir / "status.json").read_text())["status"] == "pending"
    metadata = json.loads((job_dir / "metadata.json").read_text())
    assert metadata["screening_passed_tiers"] is True
    assert metadata["screening_run_id"] == "run42"
    assert metadata["screening_random_seed"] == 42


def test_la_torre_estrecha_en_cada_tier(client):
    """Es una torre de cribado: lo que entra a un tier sale de otro más fino."""
    detalle = _cribar(client, 96)
    tiers = detalle["tiers"]

    assert [t["tier"] for t in tiers] == [0, 1, 2, 3]
    for anterior, siguiente in zip(tiers, tiers[1:]):
        assert siguiente["n_in"] == anterior["n_out"], (
            "cada tier debe recibir exactamente lo que sobrevivió al anterior"
        )
        assert siguiente["n_out"] <= siguiente["n_in"], "un tier no puede añadir"

    assert tiers[0]["kind"] == "gate"
    assert tiers[3]["kind"] == "select", "el corte final es la selección"
    assert all(t["note"] for t in tiers)


def test_el_tier_caro_solo_evalua_supervivientes(client):
    """El ahorro de la cascada: el MLFF cuesta ~0.5 s por candidato."""
    detalle = _cribar(client, 94, n=40)
    tiers = {t["tier"]: t for t in detalle["tiers"]}

    assert tiers[2]["n_in"] == tiers[1]["n_out"]
    assert tiers[2]["n_in"] <= tiers[0]["n_out"], (
        "el MLFF no debe evaluar lo que el surrogate ya descartó"
    )


def test_cada_descarte_dice_por_que(client):
    """Una torre que criba sin motivo no se puede auditar."""
    detalle = _cribar(client, 93, n=40)
    assert detalle["dropped"], "con 40 candidatos algo debería caer"

    for d in detalle["dropped"]:
        assert d["drop_reason"], d
        assert d["dropped_at_tier"] in (0, 1, 2)

    # Los supervivientes no llevan motivo de descarte.
    for item in detalle["items"]:
        assert item["dropped_at_tier"] is None


def test_la_malla_respeta_el_error_del_modelo():
    """Regresión de criterio: el surrogate tiene MAE ≈ 0.31 eV y la ventana PV
    mide 0.7 eV. Cribar por la estimación puntual tiraría materiales cuyo Eg
    real sí cae dentro."""
    import copy

    import yaml

    from buho.screening.cascade import ScreeningCascade

    cfg_path = ROOT / "config" / "generator.yaml"
    if not cfg_path.is_file():
        pytest.skip("config/generator.yaml no disponible")
    base = yaml.safe_load(cfg_path.read_text())

    fila = {"Eg_surrogate_eV": 1.81, "Eg_sigma_eV": 0.18}

    holgada = copy.deepcopy(base)
    holgada.setdefault("screening", {}).update({"sigma_k": 1.0})
    dura = copy.deepcopy(base)
    dura.setdefault("screening", {}).update({"sigma_k": 0.0})

    c_holgada = ScreeningCascade(holgada, project_root=ROOT)
    c_dura = ScreeningCascade(dura, project_root=ROOT)

    def sobrevive(casc) -> bool:
        margen = casc._sigma_k * fila["Eg_sigma_eV"]
        eg = fila["Eg_surrogate_eV"]
        return not (eg + margen < casc._pv_min or eg - margen > casc._pv_max)

    assert c_holgada._pv_max == 1.8
    assert sobrevive(c_holgada), "1.81 ± 0.18 eV sigue siendo plausible dentro de la ventana"
    assert not sobrevive(c_dura), "con malla dura, 1.81 cae por 0.01 eV"


def test_pedir_mlff_sin_dependencias_degrada(client, monkeypatch):
    """Falta torch/matgl: se desactiva el Tier 2 y se sigue, no se rompe."""
    monkeypatch.setattr(scr_svc, "_puede_importar", lambda *m: "torch" not in m)

    cfg = scr_svc.load_generator_config() if _HAY_CONFIG else {}
    tiers = {t["tier"]: t for t in scr_svc.tier_availability(cfg)}
    assert tiers[2]["available"] is False
    assert "torch" in tiers[2]["reason"]


def test_el_historial_no_arrastra_los_rankings(client):
    """Cada ejecución guarda cientos de filas; el listado debe ir ligero."""
    if not _HAY_CONFIG:
        pytest.skip("config/generator.yaml no disponible")
    client.post("/api/screening/run", json={"batch_id": 95, "n_candidates": 4})

    for item in client.get("/api/screening/runs").json()["items"]:
        assert "items" not in item
        assert {"run_id", "status", "batch_id"} <= set(item)


# ── Avance automático ────────────────────────────────────────────────────────

def test_auto_advance_esta_activo_por_defecto(tmp_path):
    """Es lo que sostiene la operación desatendida: no se cambia sin avisar."""
    from monitor_api.poller import DFTPoller

    assert DFTPoller(tmp_path, cfg={}, telegram_cfg={})._auto_advance() is True


def test_auto_advance_se_puede_apagar(tmp_path):
    """Abrir el monitor sobre un lote terminado no debería reentrenar solo."""
    from monitor_api.poller import DFTPoller

    poller = DFTPoller(tmp_path, cfg={"auto_advance": False}, telegram_cfg={})
    assert poller._auto_advance() is False


def test_con_auto_advance_apagado_no_se_lanza_nada(tmp_path, monkeypatch):
    """Regresión: un lote acabado disparaba el orquestador al abrir la app."""
    from monitor_api.poller import DFTPoller

    poller = DFTPoller(tmp_path, cfg={"auto_advance": False}, telegram_cfg={})
    poller._snapshots.update({s.job_id: s for s in SNAPSHOTS if s.status == "converged"})

    lanzados = []
    monkeypatch.setattr(poller, "_launch_runner", lambda d: lanzados.append(d))
    monkeypatch.setattr(
        poller, "_find_next_batch", lambda: (_ for _ in ()).throw(AssertionError("no debe buscarse"))
    )

    asyncio.run(poller._check_batch_done())
    assert not lanzados


def test_health_declara_si_el_monitor_puede_mutar_el_pipeline(tmp_path):
    """Que el monitor reentrene solo no puede ser una sorpresa."""
    from monitor_api import platform_caps

    assert platform_caps.describe({})["auto_advance"] is True
    assert platform_caps.describe({"auto_advance": False})["auto_advance"] is False


# ── Reportes bajo una base symlinkeada ───────────────────────────────────────

def test_reportes_funcionan_con_la_base_en_otro_volumen(tmp_path, monkeypatch):
    """Regresión: mover los datos al disco externo rompió toda la vista.

    `reports/` pasó a ser un symlink; `safe_join` sigue los enlaces con
    resolve() y, comparando contra data_root, rechazaba cada documento con un
    400. La resolución tiene que ir contra la base, no contra la raíz.
    """
    from monitor_api.services import files as files_svc

    externo = tmp_path / "otro-volumen" / "reports"
    externo.mkdir(parents=True)
    (externo / "informe.md").write_text("# Informe\n\ncontenido real\n")

    raiz = tmp_path / "proyecto"
    raiz.mkdir()
    (raiz / "reports").symlink_to(externo)
    paths.set_data_root(raiz)

    assert (raiz / "reports").is_symlink(), "el test debe probar el caso symlink"

    contenido, nombre = files_svc.read_report("reports/informe.md")
    assert "contenido real" in contenido
    assert nombre == "informe.md"


def test_la_lista_de_reportes_produce_rutas_que_se_pueden_leer(client):
    """El contrato entre los dos endpoints: lo que lista uno lo abre el otro."""
    docs = client.get("/api/reports").json()["documents"]
    if not docs:
        pytest.skip("sin reportes en esta máquina")

    for doc in docs:
        r = client.get("/api/reports/document", params={"path": doc["path"]})
        assert r.status_code == 200, f"{doc['path']} → {r.status_code}"
        assert r.json()["content"]


def test_el_traversal_sigue_bloqueado_tras_el_arreglo(client):
    """Resolver contra la base no puede abrir la puerta a salir de ella."""
    for mala in (
        "pyproject.toml",
        "reports/../pyproject.toml",
        "reports/../../etc/passwd",
        "../etc/passwd",
        "src/monitor_api/security.py",
        "/etc/passwd",
    ):
        code = client.get("/api/reports/document", params={"path": mala}).status_code
        assert code in (400, 404), f"{mala} devolvió {code}"


def test_un_symlink_que_se_escapa_de_la_base_si_se_bloquea(tmp_path):
    """La base puede ser un enlace; un enlace DENTRO que salga fuera, no."""
    from monitor_api.services import files as files_svc

    raiz = tmp_path / "proyecto"
    (raiz / "reports").mkdir(parents=True)
    secreto = tmp_path / "fuera" / "secreto.md"
    secreto.parent.mkdir()
    secreto.write_text("no deberías ver esto")
    (raiz / "reports" / "fuga.md").symlink_to(secreto)
    paths.set_data_root(raiz)

    with pytest.raises(files_svc.UnsafePathError):
        files_svc.read_report("reports/fuga.md")


# ── Configuración del agente ─────────────────────────────────────────────────

def test_models_dir_sigue_a_la_raiz_de_datos(tmp_path):
    """Un default de dataclass se evalúa al importar y congelaría data_root."""
    from monitor_api.services.agent.config import AgentConfig

    paths.set_data_root(tmp_path)
    assert AgentConfig().models_dir == tmp_path.resolve() / "models" / "ollama"


def test_no_hay_ruta_personal_por_defecto():
    """Una absoluta del autor viajaría dentro del binario y no existiría fuera."""
    from monitor_api.services.agent.config import AgentConfig, load_agent_config

    assert AgentConfig().revive_repo is None
    assert load_agent_config({}).revive_repo is None

    cfg = load_agent_config({"monitor": {"agent": {"revive_repo": "/opt/revive"}}})
    assert cfg.revive_repo == Path("/opt/revive")


def test_sin_revive_repo_el_error_dice_que_configurar(tmp_path):
    from monitor_api.services.agent.config import load_agent_config
    from monitor_api.services.agent.runtime import ensure_managed_ollama

    cfg = load_agent_config(
        {"monitor": {"agent": {"enabled": True, "manage_service": True,
                               "base_url": "http://127.0.0.1:1"}}}
    )
    with pytest.raises(RuntimeError, match="revive_repo"):
        ensure_managed_ollama(cfg, data_root=tmp_path, wait_sec=0.1)


def test_el_codigo_no_lleva_rutas_personales():
    """Guarda para que no vuelva a colarse una absoluta del autor."""
    import re

    patron = re.compile(r'"/(home|Users)/[a-z]', re.IGNORECASE)
    culpables = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in sorted((ROOT / "src" / "monitor_api").rglob("*.py"))
        for i, linea in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if patron.search(linea)
    ]
    assert not culpables, "rutas absolutas de una máquina concreta:\n  " + "\n  ".join(culpables)


def test_el_log_se_encuentra_aunque_el_job_este_en_otro_lote(tmp_path, monkeypatch):
    """`runs_dir` se fija al arrancar; el runner puede estar en otro lote.

    El detalle y el log daban 404 para todo lo que se estuviera calculando.
    """
    from monitor_api import paths
    from monitor_api.services.jobs import resolve_job_dir

    paths.set_data_root(tmp_path)
    raiz = tmp_path / "phase2_force"
    vigilado = raiz / "batch_000"
    otro = raiz / "batch_999"
    (vigilado / "propio").mkdir(parents=True)
    (otro / "ajeno").mkdir(parents=True)

    assert resolve_job_dir(vigilado, "propio") == (vigilado / "propio").resolve()
    assert resolve_job_dir(vigilado, "ajeno") == (otro / "ajeno").resolve()
    assert resolve_job_dir(vigilado, "inexistente") is None


@pytest.mark.parametrize("hostil", ["../../etc/passwd", "..", "a/b", ""])
def test_la_busqueda_ampliada_sigue_bloqueando_ids_hostiles(tmp_path, hostil):
    """Ampliar la búsqueda no puede abrir la puerta a salir del árbol."""
    from monitor_api.services.jobs import UnsafeJobIdError, resolve_job_dir

    raiz = tmp_path / "phase2_force" / "batch_000"
    raiz.mkdir(parents=True)
    with pytest.raises(UnsafeJobIdError):
        resolve_job_dir(raiz, hostil)


# ── Persistencia del cribado ─────────────────────────────────────────────────

def _run_terminado(**extra):
    """Una ejecución de cribado ya acabada, como la deja `_ejecutar`."""
    from monitor_api.services.screening import ScreeningRun

    datos = dict(
        run_id="abc123def456",
        batch_id=7,
        n_requested=300,
        use_mlff=False,
        status="done",
        stage="listo",
        n_selected=2,
        selected_candidate_ids=["c-1", "c-2"],
        selected_candidates=[{"candidate_id": "c-1"}, {"candidate_id": "c-2"}],
        finished_at=time.time(),
    )
    datos.update(extra)
    return ScreeningRun(**datos)


def test_el_cribado_sobrevive_a_cerrar_la_app(tmp_path):
    """Los seleccionados vivían solo en RAM y se perdían al cerrar.

    No es solo rehacer trabajo: `start_dft_for_run` necesita
    `selected_candidates` enteros, así que sin persistirlos un cribado no
    sobrevivía ni para lanzar el DFT que lo justificaba.
    """
    from monitor_api.services import screening

    paths.set_data_root(tmp_path)
    screening.reset_runs()
    screening._guardar(_run_terminado())

    screening.reset_runs()  # equivale a reabrir la app
    recuperado = screening.get_run("abc123def456")

    assert recuperado is not None
    assert recuperado.n_selected == 2
    assert recuperado.selected_candidates == [
        {"candidate_id": "c-1"},
        {"candidate_id": "c-2"},
    ]


def test_un_cribado_a_medias_no_se_persiste(tmp_path):
    """Solo interesa el resultado; un run vivo no sobrevive al proceso."""
    from monitor_api.services import screening

    paths.set_data_root(tmp_path)
    screening.reset_runs()
    screening._guardar(_run_terminado(status="running", stage="tier 0"))

    screening.reset_runs()
    assert screening.get_run("abc123def456") is None


def test_un_cribado_corrupto_no_tumba_el_resto_del_historial(tmp_path):
    """Un JSON truncado no puede dejar ilegible todo lo demás."""
    from monitor_api.services import screening

    paths.set_data_root(tmp_path)
    screening.reset_runs()
    screening._guardar(_run_terminado())
    (screening._dir_historial() / "roto.json").write_text("{no es json", encoding="utf-8")

    screening.reset_runs()
    assert screening.get_run("abc123def456") is not None


def test_el_historial_persistido_se_recorta(tmp_path, monkeypatch):
    """MAX_RUNS acota la memoria; también tiene que acotar el disco."""
    from monitor_api.services import screening

    paths.set_data_root(tmp_path)
    monkeypatch.setattr(screening, "MAX_RUNS", 3)
    screening.reset_runs()
    for i in range(6):
        screening._guardar(_run_terminado(run_id=f"run{i:03d}", started_at=1000.0 + i))

    screening.reset_runs()
    assert len(screening.list_runs()) == 3
    assert len(list(screening._dir_historial().glob("*.json"))) == 3
    # Se quedan los más recientes.
    assert screening.get_run("run005") is not None
    assert screening.get_run("run000") is None
