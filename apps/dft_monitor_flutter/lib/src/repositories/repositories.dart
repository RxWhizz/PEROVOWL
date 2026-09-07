import '../models/parity.dart';
import '../models/activity.dart';
import '../models/bench.dart';

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/api_client.dart';
import '../api/providers.dart';
import '../models/models.dart';

final refreshTickProvider = StreamProvider.autoDispose<int>((ref) {
  return Stream.periodic(const Duration(seconds: 5), (tick) => tick);
});

/// Reloj para lo que no cambia entre pestañeos.
///
/// Diecisiete providers colgaban del reloj de 5 s y disparaban sus peticiones
/// a la vez; una de ellas (`/api/system`) bloquea 0.4 s en el servidor. Las
/// estructuras, los candidatos, la metadata de un job y la salud del agente no
/// cambian a ese ritmo: sondearlas cada cinco segundos era trabajo tirado.
final slowRefreshTickProvider = StreamProvider.autoDispose<int>((ref) {
  return Stream.periodic(const Duration(seconds: 30), (tick) => tick);
});

final healthProvider = FutureProvider.autoDispose<Health>((ref) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return Health.fromJson(await api.getMap('/api/health'));
});

/// Sondeo de la maquina para el arranque rapido.
///
/// No cuelga de ningun reloj: medir cuesta ~2 s de CPU y el hardware no cambia
/// entre pestaneos. Se refresca a mano con `ref.invalidate`.
final hardwareProbeProvider =
    FutureProvider.autoDispose<Map<String, dynamic>>((ref) async {
  final api = ref.watch(apiClientProvider);
  return api.getMap('/api/quickstart/probe');
});

/// Estado del ultimo arranque rapido y del protocolo.
final quickStartStatusProvider =
    FutureProvider.autoDispose<Map<String, dynamic>>((ref) async {
  ref.watch(slowRefreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return api.getMap('/api/quickstart/status');
});

final authStateProvider = FutureProvider.autoDispose<AuthState>((ref) async {
  final api = ref.watch(apiClientProvider);
  return AuthState.fromJson(await api.getMap('/auth/me'));
});

class AuthActions {
  const AuthActions(this.api);

  final ApiClient api;

  Future<AuthState> login(String token) async {
    return AuthState.fromJson(
      await api.postMap('/auth/login', body: {'token': token}),
    );
  }

  Future<AuthState> logout() async {
    final state = AuthState.fromJson(await api.postMap('/auth/logout'));
    api.clearSession();
    return state;
  }
}

final authActionsProvider = Provider<AuthActions>((ref) {
  return AuthActions(ref.watch(apiClientProvider));
});

final summaryProvider = FutureProvider.autoDispose<Summary>((ref) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return Summary.fromJson(await api.getMap('/api/summary'));
});

final systemMetricsProvider = FutureProvider.autoDispose<SystemMetrics>((
  ref,
) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return SystemMetrics.fromJson(await api.getMap('/api/system'));
});

final systemHistoryProvider = FutureProvider.autoDispose<MetricsHistory>((
  ref,
) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return MetricsHistory.fromJson(
    await api.getMap('/api/system/history', query: {'minutes': 10}),
  );
});

final batchesProvider = FutureProvider.autoDispose<List<BatchInfo>>((
  ref,
) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  final body = await api.getMap('/api/batches');
  return [
    for (final item in (body['items'] as List? ?? const []))
      BatchInfo.fromJson(Map<String, dynamic>.from(item as Map)),
  ];
});

class JobsQuery {
  const JobsQuery({
    this.status,
    this.q,
    this.sort = 'formula',
    this.desc = false,
    this.limit = 250,
    this.offset = 0,
  });

  final String? status;
  final String? q;
  final String sort;
  final bool desc;
  final int limit;
  final int offset;

  Map<String, dynamic> toQuery() {
    return {
      if (status != null && status!.isNotEmpty) 'status': status,
      if (q != null && q!.isNotEmpty) 'q': q,
      'sort': sort,
      'desc': desc,
      'limit': limit,
      'offset': offset,
    };
  }

  @override
  bool operator ==(Object other) {
    return other is JobsQuery &&
        other.status == status &&
        other.q == q &&
        other.sort == sort &&
        other.desc == desc &&
        other.limit == limit &&
        other.offset == offset;
  }

  @override
  int get hashCode => Object.hash(status, q, sort, desc, limit, offset);
}

final jobsProvider = FutureProvider.autoDispose.family<JobPage, JobsQuery>((
  ref,
  query,
) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return JobPage.fromJson(
    await api.getMap('/api/jobs', query: query.toQuery()),
  );
});

final activeJobsProvider = FutureProvider.autoDispose<JobPage>((ref) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return JobPage.fromJson(
    await api.getMap(
      '/api/jobs',
      query: const JobsQuery(
        status: 'running,stalled,oscillating',
        sort: 'elapsed_min',
        desc: true,
        limit: 50,
      ).toQuery(),
    ),
  );
});

final jobProvider = FutureProvider.autoDispose.family<Job, String>((
  ref,
  jobId,
) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return Job.fromJson(await api.getMap('/api/jobs/$jobId'));
});

final jobTracesProvider = FutureProvider.autoDispose
    .family<Map<String, dynamic>, String>((ref, jobId) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return api.getMap('/api/jobs/$jobId/traces');
});

final jobMetadataProvider = FutureProvider.autoDispose
    .family<Map<String, dynamic>, String>((ref, jobId) async {
  ref.watch(slowRefreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return api.getMap('/api/jobs/$jobId/metadata');
});

final structuresProvider = FutureProvider.autoDispose<List<StructureItem>>((
  ref,
) async {
  ref.watch(slowRefreshTickProvider);
  final api = ref.watch(apiClientProvider);
  final body = await api.getMap('/api/structures');
  return [
    for (final item in (body['items'] as List? ?? const []))
      StructureItem.fromJson(Map<String, dynamic>.from(item as Map)),
  ];
});

final structureContentProvider = FutureProvider.autoDispose
    .family<StructureContent, String>((ref, id) async {
  final api = ref.watch(apiClientProvider);
  return StructureContent.fromJson(
    await api.getMap('/api/structures/content', query: {'id': id}),
  );
});

final screeningConfigProvider = FutureProvider.autoDispose<ScreeningConfig>((
  ref,
) async {
  final api = ref.watch(apiClientProvider);
  return ScreeningConfig.fromJson(await api.getMap('/api/screening/config'));
});

final screeningRunsProvider = FutureProvider.autoDispose<List<ScreeningRun>>((
  ref,
) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  final body = await api.getMap('/api/screening/runs');
  return [
    for (final item in (body['items'] as List? ?? const []))
      ScreeningRun.fromJson(Map<String, dynamic>.from(item as Map)),
  ];
});

final screeningRunProvider =
    FutureProvider.autoDispose.family<ScreeningRun, String>((ref, runId) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return ScreeningRun.fromJson(
    await api.getMap('/api/screening/runs/$runId'),
  );
});

class ScreeningActions {
  const ScreeningActions(this.api);

  final ApiClient api;

  Future<ScreeningRun> run({
    required int randomSeed,
    required int nBatches,
    required int nCandidates,
  }) async {
    final body = await api.postMap(
      '/api/screening/run',
      body: {
        'random_seed': randomSeed,
        'n_batches': nBatches,
        'n_candidates': nCandidates,
      },
    );
    return ScreeningRun.fromJson(body);
  }

  Future<ScreeningStartDftResult> startDft(
    String runId, {
    required bool startRunner,
  }) async {
    final body = await api.postMap(
      '/api/screening/runs/$runId/start-dft',
      body: {'start_runner': startRunner},
    );
    return ScreeningStartDftResult.fromJson(body);
  }
}

final screeningActionsProvider = Provider<ScreeningActions>((ref) {
  return ScreeningActions(ref.watch(apiClientProvider));
});

final discoveryStatusProvider = FutureProvider.autoDispose<DiscoveryStatus>((
  ref,
) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return DiscoveryStatus.fromJson(await api.getMap('/api/discovery/status'));
});

final discoveryConfigProvider =
    FutureProvider.autoDispose<DiscoverySpaceConfig>((
  ref,
) async {
  final api = ref.watch(apiClientProvider);
  return DiscoverySpaceConfig.fromJson(
      await api.getMap('/api/discovery/config'));
});

class DiscoveryActions {
  const DiscoveryActions(this.api);

  final ApiClient api;

  Future<DiscoveryStatus> init({bool reset = false}) async {
    return DiscoveryStatus.fromJson(
      await api.postMap('/api/discovery/init', body: {'reset': reset}),
    );
  }

  Future<DiscoveryStatus> run({
    required bool startRunner,
    required bool dryRun,
    bool? useMlff,
    int? maxRounds,
  }) async {
    return DiscoveryStatus.fromJson(
      await api.postMap(
        '/api/discovery/run',
        body: {
          'start_runner': startRunner,
          'dry_run': dryRun,
          if (useMlff != null) 'use_mlff': useMlff,
          if (maxRounds != null) 'max_rounds': maxRounds,
        },
      ),
    );
  }

  Future<DiscoveryStatus> pause() async {
    return DiscoveryStatus.fromJson(await api.postMap('/api/discovery/pause'));
  }

  Future<DiscoveryStatus> resume() async {
    return DiscoveryStatus.fromJson(await api.postMap('/api/discovery/resume'));
  }

  Future<DiscoverySpaceConfig> previewConfig(
      DiscoverySpaceConfig config) async {
    return DiscoverySpaceConfig.fromJson(
      await api.postMap('/api/discovery/config/preview', body: config.toJson()),
    );
  }

  Future<DiscoverySpaceConfig> saveConfig(DiscoverySpaceConfig config) async {
    return DiscoverySpaceConfig.fromJson(
      await api.postMap('/api/discovery/config', body: config.toJson()),
    );
  }

  Future<DiscoveryExport> export() async {
    return DiscoveryExport.fromJson(await api.postMap('/api/discovery/export'));
  }
}

final discoveryActionsProvider = Provider<DiscoveryActions>((ref) {
  return DiscoveryActions(ref.watch(apiClientProvider));
});

/// Capacidades del entorno.
///
/// `fast` omite la sonda MLFF, que lanza un proceso (y en Windows, una distro
/// de WSL): la pantalla pide primero la versión rápida para pintar algo ya, y
/// luego la completa.
final setupStatusProvider =
    FutureProvider.autoDispose.family<SetupStatus, bool>((ref, fast) async {
  final api = ref.watch(apiClientProvider);
  return SetupStatus.fromJson(
    await api.getMap('/api/setup/status', query: {'fast': fast}),
  );
});

/// Log de la instalación en curso. Se refresca solo mientras haya una viva.
final setupJobProvider = FutureProvider.autoDispose<SetupJob>((ref) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return SetupJob.fromJson(await api.getMap('/api/setup/job'));
});

class SetupActions {
  const SetupActions(this.api);

  final ApiClient api;

  Future<Map<String, dynamic>> plan(String target, {bool cuda = false}) {
    return api.postMap('/api/setup/plan',
        body: {'target': target, 'cuda': cuda});
  }

  Future<SetupJob> install(
    String target, {
    bool cuda = false,
    bool recreate = false,
  }) async {
    return SetupJob.fromJson(
      await api.postMap('/api/setup/install', body: {
        'target': target,
        'cuda': cuda,
        'recreate': recreate,
      }),
    );
  }
}

final setupActionsProvider = Provider<SetupActions>((ref) {
  return SetupActions(ref.watch(apiClientProvider));
});

class CandidatesQuery {
  const CandidatesQuery({
    this.q,
    this.generationMode,
    this.bFamily,
    this.halide,
    this.sort = 'score',
    this.desc = true,
    this.limit = 2000,
    this.offset = 0,
  });

  final String? q;
  final String? generationMode;
  final String? bFamily;
  final String? halide;
  final String sort;
  final bool desc;
  final int limit;
  final int offset;

  Map<String, dynamic> toQuery() {
    return {
      if (q != null && q!.isNotEmpty) 'q': q,
      if (generationMode != null && generationMode!.isNotEmpty)
        'generation_mode': generationMode,
      if (bFamily != null && bFamily!.isNotEmpty) 'b_family': bFamily,
      if (halide != null && halide!.isNotEmpty) 'halide': halide,
      'sort': sort,
      'desc': desc,
      'limit': limit,
      'offset': offset,
    };
  }

  @override
  bool operator ==(Object other) {
    return other is CandidatesQuery &&
        other.q == q &&
        other.generationMode == generationMode &&
        other.bFamily == bFamily &&
        other.halide == halide &&
        other.sort == sort &&
        other.desc == desc &&
        other.limit == limit &&
        other.offset == offset;
  }

  @override
  int get hashCode => Object.hash(
        q,
        generationMode,
        bFamily,
        halide,
        sort,
        desc,
        limit,
        offset,
      );
}

final candidatesProvider = FutureProvider.autoDispose
    .family<CandidatePage, CandidatesQuery>((ref, query) async {
  ref.watch(slowRefreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return CandidatePage.fromJson(
    await api.getMap('/api/candidates', query: query.toQuery()),
  );
});

final modelsProvider = FutureProvider.autoDispose<ModelsResponse>((ref) async {
  final api = ref.watch(apiClientProvider);
  return ModelsResponse.fromJson(await api.getMap('/api/models'));
});

final top8Provider = FutureProvider.autoDispose<Top8Response>((ref) async {
  final api = ref.watch(apiClientProvider);
  return Top8Response.fromJson(await api.getMap('/api/ml/top8'));
});

class MlActions {
  const MlActions(this.api);

  final ApiClient api;

  Future<Prediction> predict({
    required String a,
    required String b,
    required String x,
  }) async {
    return Prediction.fromJson(
      await api.postMap('/api/ml/predict', body: {'A': a, 'B': b, 'X': x}),
    );
  }
}

final mlActionsProvider = Provider<MlActions>((ref) {
  return MlActions(ref.watch(apiClientProvider));
});

final reportsProvider = FutureProvider.autoDispose<ReportsResponse>((
  ref,
) async {
  final api = ref.watch(apiClientProvider);
  return ReportsResponse.fromJson(await api.getMap('/api/reports'));
});

final reportDocumentProvider = FutureProvider.autoDispose
    .family<ReportDocument, String>((ref, path) async {
  final api = ref.watch(apiClientProvider);
  return ReportDocument.fromJson(
    await api.getMap('/api/reports/document', query: {'path': path}),
  );
});

final reportFigureUrlProvider = Provider.family<String, String>((ref, path) {
  final api = ref.watch(apiClientProvider);
  return api.url('/api/reports/figure', query: {'path': path});
});

final agentHealthProvider = FutureProvider.autoDispose<AgentHealth>((
  ref,
) async {
  ref.watch(slowRefreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return AgentHealth.fromJson(await api.getMap('/api/agent/health'));
});

class AgentActions {
  const AgentActions(this.api);

  final ApiClient api;

  Future<AgentChatResponse> chat({
    required String message,
    required List<Map<String, String>> history,
    required bool structured,
    String? jobId,
  }) async {
    return AgentChatResponse.fromJson(
      await api.postMap(
        '/api/agent/chat',
        body: {
          'message': message,
          'history': history,
          'structured': structured,
          if (jobId != null && jobId.isNotEmpty) 'job_id': jobId,
        },
      ),
    );
  }

  Future<AgentProposal> approve(String id) async {
    return AgentProposal.fromJson(
      await api.postMap('/api/agent/proposals/$id/approve'),
    );
  }

  Future<AgentProposal> reject(String id) async {
    return AgentProposal.fromJson(
      await api.postMap('/api/agent/proposals/$id/reject'),
    );
  }
}

final agentActionsProvider = Provider<AgentActions>((ref) {
  return AgentActions(ref.watch(apiClientProvider));
});

class JobActions {
  const JobActions(this.api);

  final ApiClient api;

  Future<void> kill(String jobId) async {
    await api.postMap('/api/jobs/$jobId/kill');
  }

  Future<void> retry(String jobId) async {
    await api.postMap('/api/jobs/$jobId/retry');
  }

  Future<void> startBatch(int batchId) async {
    await api.postMap('/api/batches/$batchId/start');
  }
}

final jobActionsProvider = Provider<JobActions>((ref) {
  return JobActions(ref.watch(apiClientProvider));
});

// ── Calibración de rendimiento ───────────────────────────────────────────────

/// Estado del barrido. En reposo basta el reloj lento; mientras mide, el rápido.
final benchStatusProvider = FutureProvider.autoDispose<BenchStatus>((
  ref,
) async {
  final previo = ref.state.valueOrNull;
  if (previo?.running == true) {
    ref.watch(refreshTickProvider);
  } else {
    ref.watch(slowRefreshTickProvider);
  }
  final api = ref.watch(apiClientProvider);
  return BenchStatus.fromJson(await api.getMap('/api/bench'));
});

class BenchActions {
  BenchActions(this.api);

  final ApiClient api;

  Future<void> run({required String mode, bool force = false}) async {
    await api.postMap('/api/bench/run', body: {'mode': mode, 'force': force});
  }

  Future<void> cancel() async {
    await api.postMap('/api/bench/cancel');
  }
}

final benchActionsProvider = Provider<BenchActions>((ref) {
  return BenchActions(ref.watch(apiClientProvider));
});

// ── Actividad del sistema ────────────────────────────────────────────────────

final activityProvider = FutureProvider.autoDispose<Activity>((ref) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return Activity.fromJson(await api.getMap('/api/activity'));
});

/// Paridad del lote: predicho contra el DFT ya calculado.
///
/// Leer decenas de logs de GPAW cuesta, y solo cambia cuando termina un job:
/// va en el reloj lento.
final parityProvider = FutureProvider.autoDispose<ParityData>((ref) async {
  ref.watch(slowRefreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return ParityData.fromJson(
    await api.getMap('/api/ml/parity', query: {'limit': 50}),
  );
});

// ── Log en vivo ──────────────────────────────────────────────────────────────

/// Qué log se pide: trabajo y, opcionalmente, cuál de sus ficheros.
///
/// Es una clase con `==` propio para que Riverpod reutilice el provider cuando
/// la petición no cambia; con un record anónimo cada `build` crearía una
/// familia nueva y el log se recargaría desde cero en cada fotograma.
class JobLogPeticion {
  const JobLogPeticion({required this.jobId, this.etiqueta});

  final String jobId;
  final String? etiqueta;

  @override
  bool operator ==(Object other) =>
      other is JobLogPeticion &&
      other.jobId == jobId &&
      other.etiqueta == etiqueta;

  @override
  int get hashCode => Object.hash(jobId, etiqueta);
}

final jobLogProvider =
    FutureProvider.autoDispose.family<LogJob, JobLogPeticion>((ref, p) async {
  ref.watch(refreshTickProvider);
  final api = ref.watch(apiClientProvider);
  return LogJob.fromJson(
    await api.getMap(
      '/api/jobs/${p.jobId}/log',
      query: {'tail': 400, if (p.etiqueta != null) 'label': p.etiqueta},
    ),
  );
});
