String? _textOrNull(Object? value) {
  if (value == null) return null;
  final text = '$value';
  return text.isEmpty ? null : text;
}

String _textOrDash(Object? value) => _textOrNull(value) ?? '-';

double? _doubleOrNull(Object? value) =>
    value is num ? value.toDouble() : double.tryParse('$value');

int _intOrZero(Object? value) {
  if (value is num) return value.toInt();
  return int.tryParse('$value') ?? 0;
}

Map<String, dynamic> _mapOrEmpty(Object? value) {
  if (value is Map) return Map<String, dynamic>.from(value);
  return {};
}

Map<String, int> _intMap(Object? value) {
  if (value is! Map) return {};
  return {
    for (final entry in Map<String, dynamic>.from(value).entries)
      entry.key: _intOrZero(entry.value),
  };
}

List<String> _stringList(Object? value) {
  if (value is! List) return const [];
  return [for (final item in value) _textOrDash(item)]
      .where((item) => item != '-')
      .toList();
}

List<double> _doubleList(Object? value) {
  if (value is! List) return const [];
  return [
    for (final item in value)
      if (_doubleOrNull(item) != null) _doubleOrNull(item)!
  ];
}

Map<String, bool> _boolMap(Object? value) {
  if (value is! Map) return const {};
  return {
    for (final entry in Map<String, dynamic>.from(value).entries)
      entry.key: entry.value == true,
  };
}

Map<String, List<String>> _stringListMap(Object? value) {
  if (value is! Map) return const {};
  return {
    for (final entry in Map<String, dynamic>.from(value).entries)
      entry.key: _stringList(entry.value),
  };
}

class Health {
  const Health({
    required this.ok,
    required this.version,
    required this.dataRoot,
    required this.runsDir,
    required this.runsMounted,
    required this.runsState,
    required this.nJobsTracked,
    required this.lastPollAgeSec,
    required this.wsClients,
    required this.autoAdvance,
  });

  factory Health.fromJson(Map<String, dynamic> json) {
    final paths = Map<String, dynamic>.from(json['paths'] as Map? ?? {});
    return Health(
      ok: json['ok'] == true,
      version: json['version'] as String? ?? '-',
      dataRoot: paths['data_root'] as String? ?? '-',
      runsDir: json['runs_dir'] as String? ?? '-',
      runsMounted: json['runs_mounted'] == true,
      // Un motor viejo no manda runs_state: se deduce del booleano de siempre.
      runsState: json['runs_state'] as String? ??
          (json['runs_mounted'] == true ? 'montado' : 'desmontado'),
      nJobsTracked: (json['n_jobs_tracked'] as num? ?? 0).toInt(),
      lastPollAgeSec: (json['last_poll_age_sec'] as num?)?.toDouble(),
      wsClients: (json['ws_clients'] as num? ?? 0).toInt(),
      autoAdvance: Map<String, dynamic>.from(
            json['platform'] as Map? ?? {},
          )['auto_advance'] ==
          true,
    );
  }

  final bool ok;
  final String version;
  final String dataRoot;
  final String runsDir;
  final bool runsMounted;

  /// `montado` | `sin_estrenar` | `desmontado`. Solo el último es un fallo:
  /// en una instalación recién hecha `runs/` aún no existe y eso es normal.
  final String runsState;
  final int nJobsTracked;
  final double? lastPollAgeSec;
  final int wsClients;

  /// Si el monitor puede mutar el pipeline por su cuenta al detectar un lote
  /// acabado (relanzar runner, disparar el orquestador de active learning).
  final bool autoAdvance;
}

class AuthState {
  const AuthState({required this.authenticated, required this.authEnabled});

  factory AuthState.fromJson(Map<String, dynamic> json) {
    return AuthState(
      authenticated: json['authenticated'] == true,
      authEnabled: json['auth_enabled'] == true,
    );
  }

  final bool authenticated;
  final bool authEnabled;
}

class Summary {
  const Summary({
    required this.total,
    required this.pending,
    required this.running,
    required this.converged,
    required this.failed,
    required this.stalled,
    required this.oscillating,
    required this.skippedDuplicate,
    required this.convergenceRate,
  });

  factory Summary.fromJson(Map<String, dynamic> json) {
    return Summary(
      total: (json['total'] as num? ?? 0).toInt(),
      pending: (json['n_pending'] as num? ?? 0).toInt(),
      running: (json['n_running'] as num? ?? 0).toInt(),
      converged: (json['n_converged'] as num? ?? 0).toInt(),
      failed: (json['n_failed'] as num? ?? 0).toInt(),
      stalled: (json['n_stalled'] as num? ?? 0).toInt(),
      oscillating: (json['n_oscillating'] as num? ?? 0).toInt(),
      skippedDuplicate: (json['n_skipped_duplicate'] as num? ?? 0).toInt(),
      convergenceRate: (json['convergence_rate'] as num?)?.toDouble(),
    );
  }

  final int total;
  final int pending;
  final int running;
  final int converged;
  final int failed;
  final int stalled;
  final int oscillating;
  final int skippedDuplicate;
  final double? convergenceRate;
}

class SystemMetrics {
  const SystemMetrics({
    required this.cpuPercent,
    required this.cpuPerCore,
    required this.ramUsedGb,
    required this.ramTotalGb,
    required this.ramPercent,
    required this.gpuTemps,
    required this.coreTempMax,
    this.nvmeTemp,
  });

  factory SystemMetrics.fromJson(Map<String, dynamic> json) {
    return SystemMetrics(
      cpuPercent: (json['cpu_percent'] as num? ?? 0).toDouble(),
      cpuPerCore: [
        for (final value in (json['cpu_per_core'] as List? ?? const []))
          if (value is num) value.toDouble(),
      ],
      ramUsedGb: (json['ram_used_gb'] as num? ?? 0).toDouble(),
      ramTotalGb: (json['ram_total_gb'] as num? ?? 0).toDouble(),
      ramPercent: (json['ram_percent'] as num? ?? 0).toDouble(),
      gpuTemps: [
        for (final value in (json['gpu_temps'] as List? ?? const []))
          if (value is num) value.toDouble(),
      ],
      coreTempMax: (json['core_temp_max'] as num? ?? 0).toDouble(),
      nvmeTemp: (json['nvme_temp'] as num?)?.toDouble(),
    );
  }

  final double cpuPercent;
  final List<double> cpuPerCore;
  final double ramUsedGb;
  final double ramTotalGb;
  final double ramPercent;
  final List<double> gpuTemps;
  final double coreTempMax;
  final double? nvmeTemp;
}

class MetricsSample {
  const MetricsSample({
    required this.t,
    required this.cpuPercent,
    required this.ramPercent,
    required this.ramUsedGb,
    required this.coreTempMax,
    this.gpuTempMax,
  });

  factory MetricsSample.fromJson(Map<String, dynamic> json) {
    return MetricsSample(
      t: (json['t'] as num? ?? 0).toDouble(),
      cpuPercent: (json['cpu_percent'] as num? ?? 0).toDouble(),
      ramPercent: (json['ram_percent'] as num? ?? 0).toDouble(),
      ramUsedGb: (json['ram_used_gb'] as num? ?? 0).toDouble(),
      coreTempMax: (json['core_temp_max'] as num? ?? 0).toDouble(),
      gpuTempMax: (json['gpu_temp_max'] as num?)?.toDouble(),
    );
  }

  final double t;
  final double cpuPercent;
  final double ramPercent;
  final double ramUsedGb;
  final double coreTempMax;
  final double? gpuTempMax;
}

class MetricsHistory {
  const MetricsHistory({required this.samples, required this.intervalSec});

  factory MetricsHistory.fromJson(Map<String, dynamic> json) {
    return MetricsHistory(
      samples: [
        for (final item in (json['samples'] as List? ?? const []))
          MetricsSample.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      intervalSec: (json['interval_sec'] as num? ?? 0).toInt(),
    );
  }

  final List<MetricsSample> samples;
  final int intervalSec;
}

class Job {
  const Job({
    required this.jobId,
    required this.formula,
    required this.status,
    required this.elapsedMin,
    required this.pid,
    required this.mpiCores,
  });

  factory Job.fromJson(Map<String, dynamic> json) {
    return Job(
      jobId: json['job_id'] as String? ?? '',
      formula: json['formula'] as String? ?? '',
      status: json['status'] as String? ?? 'unknown',
      elapsedMin: (json['elapsed_min'] as num?)?.toDouble(),
      pid: (json['pid'] as num?)?.toInt(),
      mpiCores: (json['mpi_cores'] as num?)?.toInt(),
    );
  }

  final String jobId;
  final String formula;
  final String status;
  final double? elapsedMin;
  final int? pid;
  final int? mpiCores;
}

class JobPage {
  const JobPage({required this.items, required this.total});

  factory JobPage.fromJson(Map<String, dynamic> json) {
    return JobPage(
      items: [
        for (final item in (json['items'] as List? ?? const []))
          Job.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      total: (json['total'] as num? ?? 0).toInt(),
    );
  }

  final List<Job> items;
  final int total;
}

class BatchInfo {
  const BatchInfo({
    required this.batchId,
    required this.path,
    required this.counts,
    required this.total,
    required this.isCurrent,
    required this.runnerLaunched,
    required this.nPending,
    this.ratePerHour,
    this.etaSec,
  });

  factory BatchInfo.fromJson(Map<String, dynamic> json) {
    return BatchInfo(
      batchId: (json['batch_id'] as num?)?.toInt() ?? -1,
      path: json['path'] as String? ?? '',
      counts: Map<String, int>.from(
        (json['counts'] as Map? ?? {}).map(
          (key, value) => MapEntry('$key', (value as num).toInt()),
        ),
      ),
      total: (json['total'] as num? ?? 0).toInt(),
      isCurrent: json['is_current'] == true,
      runnerLaunched: json['runner_launched'] == true,
      nPending: (json['n_pending'] as num? ?? 0).toInt(),
      ratePerHour: (json['rate_per_hour'] as num?)?.toDouble(),
      etaSec: (json['eta_sec'] as num?)?.toDouble(),
    );
  }

  final int batchId;
  final String path;
  final Map<String, int> counts;
  final int total;
  final bool isCurrent;
  final bool runnerLaunched;
  final int nPending;
  final double? ratePerHour;
  final double? etaSec;
}

class StructureItem {
  const StructureItem({
    required this.id,
    required this.name,
    required this.group,
    required this.format,
    this.detail,
    this.mtime,
  });

  factory StructureItem.fromJson(Map<String, dynamic> json) {
    return StructureItem(
      id: json['id'] as String? ?? '',
      name: json['name'] as String? ?? '',
      group: json['group'] as String? ?? '',
      format: json['format'] as String? ?? '',
      detail: json['detail'] as String?,
      mtime: (json['mtime'] as num?)?.toDouble(),
    );
  }

  final String id;
  final String name;
  final String group;
  final String format;
  final String? detail;
  final double? mtime;
}

class StructureContent {
  const StructureContent({
    required this.id,
    required this.name,
    required this.content,
    required this.metadata,
  });

  factory StructureContent.fromJson(Map<String, dynamic> json) {
    return StructureContent(
      id: json['id'] as String? ?? '',
      name: json['name'] as String? ?? '',
      content: json['content'] as String? ?? '',
      metadata: Map<String, dynamic>.from(json['metadata'] as Map? ?? {}),
    );
  }

  final String id;
  final String name;
  final String content;
  final Map<String, dynamic> metadata;
}

class FunnelTier {
  const FunnelTier({
    required this.tier,
    required this.name,
    required this.kind,
    required this.nIn,
    required this.nOut,
    required this.nDropped,
    required this.ran,
  });

  factory FunnelTier.fromJson(Map<String, dynamic> json) {
    return FunnelTier(
      tier: (json['tier'] as num? ?? 0).toInt(),
      name: json['name'] as String? ?? '',
      kind: json['kind'] as String? ?? '',
      nIn: (json['n_in'] as num? ?? 0).toInt(),
      nOut: (json['n_out'] as num? ?? 0).toInt(),
      nDropped: (json['n_dropped'] as num? ?? 0).toInt(),
      ran: json['ran'] != false,
    );
  }

  final int tier;
  final String name;
  final String kind;
  final int nIn;
  final int nOut;
  final int nDropped;
  final bool ran;
}

class ScreeningConfig {
  const ScreeningConfig({
    required this.available,
    required this.reason,
    required this.gates,
  });

  factory ScreeningConfig.fromJson(Map<String, dynamic> json) {
    return ScreeningConfig(
      available: json['available'] == true,
      reason: json['reason'] as String?,
      gates: Map<String, dynamic>.from(json['gates'] as Map? ?? {}),
    );
  }

  final bool available;
  final String? reason;
  final Map<String, dynamic> gates;
}

class ScreeningRun {
  const ScreeningRun({
    required this.runId,
    required this.batchId,
    required this.status,
    required this.stage,
    required this.randomSeed,
    required this.nBatches,
    required this.nRequested,
    required this.nSelected,
    required this.tiers,
    required this.items,
    required this.nItemsTotal,
    this.error,
    this.dftBatchPath,
    this.dftPrepared,
  });

  factory ScreeningRun.fromJson(Map<String, dynamic> json) {
    return ScreeningRun(
      runId: json['run_id'] as String? ?? '',
      batchId: (json['batch_id'] as num? ?? 0).toInt(),
      status: json['status'] as String? ?? 'unknown',
      stage: json['stage'] as String? ?? '',
      randomSeed: (json['random_seed'] as num? ?? 0).toInt(),
      nBatches: (json['n_batches'] as num? ?? 1).toInt(),
      nRequested: (json['n_requested'] as num? ?? 0).toInt(),
      nSelected: (json['n_selected'] as num? ?? 0).toInt(),
      tiers: [
        for (final item in (json['tiers'] as List? ?? const []))
          FunnelTier.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      items: [
        for (final item in (json['items'] as List? ?? const []))
          Map<String, dynamic>.from(item as Map),
      ],
      nItemsTotal: (json['n_items_total'] as num? ??
              (json['items'] as List? ?? const []).length)
          .toInt(),
      error: json['error'] as String?,
      dftBatchPath: json['dft_batch_path'] as String?,
      dftPrepared: (json['dft_prepared'] as num?)?.toInt(),
    );
  }

  final String runId;
  final int batchId;
  final String status;
  final String stage;
  final int randomSeed;
  final int nBatches;
  final int nRequested;
  final int nSelected;
  final List<FunnelTier> tiers;
  final List<Map<String, dynamic>> items;
  final int nItemsTotal;
  final String? error;
  final String? dftBatchPath;
  final int? dftPrepared;
}

class ScreeningStartDftResult {
  const ScreeningStartDftResult({
    required this.runId,
    required this.batchId,
    required this.batchPath,
    required this.nPrepared,
    required this.runnerLaunched,
    this.runnerError,
  });

  factory ScreeningStartDftResult.fromJson(Map<String, dynamic> json) {
    return ScreeningStartDftResult(
      runId: json['run_id'] as String? ?? '',
      batchId: (json['batch_id'] as num? ?? 0).toInt(),
      batchPath: json['batch_path'] as String? ?? '',
      nPrepared: (json['n_prepared'] as num? ?? 0).toInt(),
      runnerLaunched: json['runner_launched'] == true,
      runnerError: json['runner_error'] as String?,
    );
  }

  final String runId;
  final int batchId;
  final String batchPath;
  final int nPrepared;
  final bool runnerLaunched;
  final String? runnerError;
}

class DiscoverySpacePreview {
  const DiscoverySpacePreview({
    required this.totalGenerated,
    required this.physicallyViable,
    required this.rejectedPhysical,
    required this.modeCounts,
  });

  factory DiscoverySpacePreview.fromJson(Map<String, dynamic> json) {
    return DiscoverySpacePreview(
      totalGenerated: _intOrZero(json['total_generated']),
      physicallyViable: _intOrZero(json['physically_viable']),
      rejectedPhysical: _intOrZero(json['rejected_physical']),
      modeCounts: _intMap(json['mode_counts']),
    );
  }

  final int totalGenerated;
  final int physicallyViable;
  final int rejectedPhysical;
  final Map<String, int> modeCounts;
}

class DiscoverySpaceConfig {
  const DiscoverySpaceConfig({
    required this.aSites,
    required this.bSites,
    required this.xSites,
    required this.modes,
    required this.minFraction,
    required this.maxFraction,
    required this.fractionStep,
    required this.fractionValues,
    required this.includeMultiMixed,
    required this.dftPerRound,
    required this.availableSpecies,
    this.preview,
    this.source,
    this.overridePath,
    this.overrideSaved = false,
  });

  factory DiscoverySpaceConfig.fromJson(Map<String, dynamic> json) {
    return DiscoverySpaceConfig(
      aSites: _stringList(json['A_sites']),
      bSites: _stringList(json['B_sites']),
      xSites: _stringList(json['X_sites']),
      modes: {
        'pure': true,
        'A_mixed': true,
        'B_mixed': true,
        'X_mixed': true,
        'multi_mixed': false,
        ..._boolMap(json['modes']),
      },
      minFraction: _doubleOrNull(json['min_fraction']) ?? 0.05,
      maxFraction: _doubleOrNull(json['max_fraction']) ?? 0.95,
      fractionStep: _doubleOrNull(json['fraction_step']) ?? 0.01,
      fractionValues: _doubleList(json['fraction_values']),
      includeMultiMixed: json['include_multi_mixed'] == true,
      dftPerRound: _intOrZero(json['dft_per_round']),
      availableSpecies: _stringListMap(json['available_species']),
      preview: json['preview'] is Map
          ? DiscoverySpacePreview.fromJson(_mapOrEmpty(json['preview']))
          : null,
      source: _textOrNull(json['source']),
      overridePath: _textOrNull(json['override_path']),
      overrideSaved: json['override_saved'] == true,
    );
  }

  final List<String> aSites;
  final List<String> bSites;
  final List<String> xSites;
  final Map<String, bool> modes;
  final double minFraction;
  final double maxFraction;
  final double fractionStep;
  final List<double> fractionValues;
  final bool includeMultiMixed;
  final int dftPerRound;
  final Map<String, List<String>> availableSpecies;
  final DiscoverySpacePreview? preview;
  final String? source;
  final String? overridePath;
  final bool overrideSaved;

  Map<String, dynamic> toJson() {
    return {
      'A_sites': aSites,
      'B_sites': bSites,
      'X_sites': xSites,
      'modes': modes,
      'min_fraction': minFraction,
      'max_fraction': maxFraction,
      'fraction_step': fractionStep,
      'include_multi_mixed': includeMultiMixed,
      'dft_per_round': dftPerRound,
    };
  }
}

class DiscoveryBackground {
  const DiscoveryBackground({required this.running, this.lastError});

  factory DiscoveryBackground.fromJson(Map<String, dynamic> json) {
    return DiscoveryBackground(
      running: json['running'] == true,
      lastError: _textOrNull(json['last_error']),
    );
  }

  final bool running;
  final String? lastError;
}

class DiscoveryState {
  const DiscoveryState({
    required this.status,
    required this.currentRound,
    required this.dftPerRound,
    required this.space,
    required this.paths,
    required this.lastScreening,
    required this.lastPrepared,
    this.stopReason,
    this.activeRoundDir,
    this.activeRunsDir,
    this.nSelectedActive,
    this.mlffWarning,
  });

  factory DiscoveryState.fromJson(Map<String, dynamic> json) {
    return DiscoveryState(
      status: _textOrDash(json['status']),
      currentRound: _intOrZero(json['current_round']),
      dftPerRound: _intOrZero(json['dft_per_round']),
      space: _mapOrEmpty(json['space']),
      paths: _mapOrEmpty(json['paths']),
      lastScreening: _mapOrEmpty(json['last_screening']),
      lastPrepared: _mapOrEmpty(json['last_prepared']),
      stopReason: _textOrNull(json['stop_reason']),
      activeRoundDir: _textOrNull(json['active_round_dir']),
      activeRunsDir: _textOrNull(json['active_runs_dir']),
      nSelectedActive: _intOrZero(json['n_selected_active']),
      mlffWarning: json['mlff_warning'] is Map
          ? _mapOrEmpty(json['mlff_warning'])
          : null,
    );
  }

  final String status;
  final int currentRound;
  final int dftPerRound;
  final Map<String, dynamic> space;
  final Map<String, dynamic> paths;
  final Map<String, dynamic> lastScreening;
  final Map<String, dynamic> lastPrepared;
  final String? stopReason;
  final String? activeRoundDir;
  final String? activeRunsDir;
  final int? nSelectedActive;

  /// Por qué la última criba corrió sin Tier 2, si es que corrió sin él.
  ///
  /// Que falte el entorno MLFF ya no mata la ronda, así que hay que decirlo en
  /// algún sitio: si no, el protocolo avanzaría con menos criba del que el
  /// usuario cree y sin que nada lo indique.
  final Map<String, dynamic>? mlffWarning;

  String? get mlffError => _textOrNull(mlffWarning?['error']);
  String? get mlffRemediacion => _textOrNull(mlffWarning?['remediation']);
}

class DiscoveryCandidate {
  const DiscoveryCandidate({
    required this.candidateId,
    required this.formula,
    required this.generationMode,
    required this.bFamily,
    required this.dominantHalide,
    required this.status,
    this.bandgapEv,
    this.bandgapSigmaEv,
    this.formationEvAtom,
    this.electronMass,
    this.holeMass,
    this.epsInf,
    this.excitonBindingMev,
    this.pvScore,
    this.acquisitionScore,
    this.uncertaintyScore,
    this.toleranceT,
    this.octFactor,
    this.roundSelected,
  });

  factory DiscoveryCandidate.fromJson(Map<String, dynamic> json) {
    return DiscoveryCandidate(
      candidateId: _textOrDash(json['candidate_id']),
      formula: _textOrDash(json['formula']),
      generationMode: _textOrDash(json['generation_mode']),
      bFamily: _textOrDash(json['B_family']),
      dominantHalide: _textOrDash(json['dominant_halide']),
      status: _textOrDash(json['status']),
      bandgapEv: _doubleOrNull(json['Eg_surrogate_eV']),
      bandgapSigmaEv: _doubleOrNull(json['Eg_sigma_eV']),
      formationEvAtom: _doubleOrNull(json['Eform_eV_atom']),
      electronMass: _doubleOrNull(json['meff_e_pred_m0']),
      holeMass: _doubleOrNull(json['meff_h_pred_m0']),
      epsInf: _doubleOrNull(json['eps_inf_pred']),
      excitonBindingMev: _doubleOrNull(json['exciton_binding_meV']),
      pvScore: _doubleOrNull(json['pv_score_ml']),
      acquisitionScore: _doubleOrNull(json['acquisition_score']),
      uncertaintyScore: _doubleOrNull(json['uncertainty_score']),
      toleranceT: _doubleOrNull(json['tolerance_t']),
      octFactor: _doubleOrNull(json['oct_factor']),
      roundSelected: _textOrNull(json['round_selected']),
    );
  }

  final String candidateId;
  final String formula;
  final String generationMode;
  final String bFamily;
  final String dominantHalide;
  final String status;
  final double? bandgapEv;
  final double? bandgapSigmaEv;
  final double? formationEvAtom;
  final double? electronMass;
  final double? holeMass;
  final double? epsInf;
  final double? excitonBindingMev;
  final double? pvScore;
  final double? acquisitionScore;
  final double? uncertaintyScore;
  final double? toleranceT;
  final double? octFactor;
  final String? roundSelected;
}

class DiscoveryStatus {
  const DiscoveryStatus({
    required this.state,
    required this.counts,
    required this.total,
    required this.seen,
    required this.percent,
    required this.frontier,
    required this.queue,
    required this.paths,
    required this.runner,
    this.config,
    this.background,
  });

  factory DiscoveryStatus.fromJson(Map<String, dynamic> json) {
    final coverage = _mapOrEmpty(json['coverage']);
    return DiscoveryStatus(
      state: DiscoveryState.fromJson(_mapOrEmpty(json['state'])),
      counts: _intMap(json['counts']),
      total: _intOrZero(coverage['total']),
      seen: _intOrZero(coverage['seen']),
      percent: _doubleOrNull(coverage['percent']) ?? 0,
      frontier: [
        for (final item in (json['frontier'] as List? ?? const []))
          DiscoveryCandidate.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      queue: [
        for (final item in (json['queue'] as List? ?? const []))
          DiscoveryCandidate.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      paths: {
        for (final entry in _mapOrEmpty(json['paths']).entries)
          entry.key: _textOrDash(entry.value),
      },
      runner: _mapOrEmpty(json['runner']),
      config: json['config'] is Map
          ? DiscoverySpaceConfig.fromJson(_mapOrEmpty(json['config']))
          : null,
      background: json['background'] is Map
          ? DiscoveryBackground.fromJson(_mapOrEmpty(json['background']))
          : null,
    );
  }

  final DiscoveryState state;
  final Map<String, int> counts;
  final int total;
  final int seen;
  final double percent;
  final List<DiscoveryCandidate> frontier;
  final List<DiscoveryCandidate> queue;
  final Map<String, String> paths;
  final Map<String, dynamic> runner;
  final DiscoverySpaceConfig? config;
  final DiscoveryBackground? background;

  bool get backgroundRunning => background?.running ?? false;
  bool get isWorking =>
      backgroundRunning ||
      state.status == 'screening' ||
      state.status == 'dft_running';

  int get selectedForDft =>
      counts['dft_selected'] ?? state.nSelectedActive ?? queue.length;

  bool get runnerStale => runner['stale'] == true;

  String? get runnerError => _textOrNull(runner['error']);
}

class DiscoveryExport {
  const DiscoveryExport({
    required this.report,
    required this.ledger,
    required this.frontier,
  });

  factory DiscoveryExport.fromJson(Map<String, dynamic> json) {
    return DiscoveryExport(
      report: _textOrDash(json['report']),
      ledger: _textOrDash(json['ledger']),
      frontier: _textOrDash(json['frontier']),
    );
  }

  final String report;
  final String ledger;
  final String frontier;
}

class RangeFilter {
  const RangeFilter({required this.min, required this.max});

  factory RangeFilter.fromJson(Map<String, dynamic> json) {
    return RangeFilter(
      min: (json['min'] as num? ?? 0).toDouble(),
      max: (json['max'] as num? ?? 0).toDouble(),
    );
  }

  final double min;
  final double max;
}

class Candidate {
  const Candidate({
    required this.candidateId,
    required this.formula,
    required this.generationMode,
    required this.toleranceT,
    required this.octFactor,
    required this.volumeA3,
    required this.score,
    required this.bFamily,
    required this.dominantHalide,
    required this.hasDft,
    required this.dftStatus,
    this.pvScore,
    this.egPred,
    this.egDft,
    this.batch,
  });

  factory Candidate.fromJson(Map<String, dynamic> json) {
    return Candidate(
      candidateId: json['candidate_id'] as String?,
      formula: json['formula'] as String?,
      generationMode: json['generation_mode'] as String?,
      toleranceT: (json['tolerance_t'] as num?)?.toDouble(),
      octFactor: (json['oct_factor'] as num?)?.toDouble(),
      volumeA3: (json['vol_est_A3'] as num?)?.toDouble(),
      score: (json['score'] as num?)?.toDouble(),
      bFamily: json['b_family'] as String?,
      dominantHalide: json['dominant_halide'] as String?,
      hasDft: json['has_dft'] == true,
      dftStatus: json['dft_status'] as String?,
      pvScore: (json['pv_score'] as num?)?.toDouble(),
      egPred: (json['eg_pred_ev'] as num?)?.toDouble(),
      egDft: (json['eg_dft_ev'] as num?)?.toDouble(),
      batch: json['batch'] as String?,
    );
  }

  /// Cercanía a la ventana fotovoltaica (gaussiana centrada en 1.45 eV).
  final double? pvScore;

  /// Bandgap que predice el modelo a partir de la composición.
  final double? egPred;

  /// Bandgap que dio el DFT, leído del log de GPAW.
  final double? egDft;

  /// Lote del que salió: los candidatos ya no vienen de un único directorio.
  final String? batch;

  final String? candidateId;
  final String? formula;
  final String? generationMode;
  final double? toleranceT;
  final double? octFactor;
  final double? volumeA3;
  final double? score;
  final String? bFamily;
  final String? dominantHalide;
  final bool hasDft;
  final String? dftStatus;
}

class CandidatePage {
  const CandidatePage({
    required this.items,
    required this.total,
    required this.source,
    required this.goldschmidt,
    required this.octahedral,
    required this.facets,
  });

  factory CandidatePage.fromJson(Map<String, dynamic> json) {
    final filters = Map<String, dynamic>.from(json['filters'] as Map? ?? {});
    return CandidatePage(
      items: [
        for (final item in (json['items'] as List? ?? const []))
          Candidate.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      total: (json['total'] as num? ?? 0).toInt(),
      source: json['source'] as String? ?? '-',
      goldschmidt: RangeFilter.fromJson(
        Map<String, dynamic>.from(filters['goldschmidt'] as Map? ?? {}),
      ),
      octahedral: RangeFilter.fromJson(
        Map<String, dynamic>.from(filters['octahedral'] as Map? ?? {}),
      ),
      facets: {
        for (final entry in Map<String, dynamic>.from(
          json['facets'] as Map? ?? {},
        ).entries)
          entry.key: [
            for (final value in (entry.value as List? ?? const [])) '$value',
          ],
      },
    );
  }

  final List<Candidate> items;
  final int total;
  final String source;
  final RangeFilter goldschmidt;
  final RangeFilter octahedral;
  final Map<String, List<String>> facets;
}

class Prediction {
  const Prediction({
    required this.material,
    required this.bandgapPred,
    required this.bandgapUncertainty,
    required this.stabilityScore,
    required this.solarScore,
    required this.inPvWindow,
    required this.modelName,
  });

  factory Prediction.fromJson(Map<String, dynamic> json) {
    return Prediction(
      material: json['material'] as String? ?? '',
      bandgapPred: (json['bandgap_pred'] as num? ?? 0).toDouble(),
      bandgapUncertainty: (json['bandgap_uncertainty'] as num? ?? 0).toDouble(),
      stabilityScore: (json['stability_score'] as num? ?? 0).toDouble(),
      solarScore: (json['solar_score'] as num? ?? 0).toDouble(),
      inPvWindow: json['in_pv_window'] == true,
      modelName: json['model_name'] as String? ?? '-',
    );
  }

  final String material;
  final double bandgapPred;
  final double bandgapUncertainty;
  final double stabilityScore;
  final double solarScore;
  final bool inPvWindow;
  final String modelName;
}

class Top8Row {
  const Top8Row({
    required this.material,
    this.egDft,
    this.egExp,
    this.egMl,
    this.egMlStd,
    this.solarScore,
    this.inPvWindow,
    this.error,
  });

  factory Top8Row.fromJson(Map<String, dynamic> json) {
    return Top8Row(
      material: json['material'] as String? ?? '',
      egDft: (json['Eg_dft_eV'] as num?)?.toDouble(),
      egExp: (json['Eg_exp_eV'] as num?)?.toDouble(),
      egMl: (json['Eg_ml_eV'] as num?)?.toDouble(),
      egMlStd: (json['Eg_ml_std_eV'] as num?)?.toDouble(),
      solarScore: (json['solar_score'] as num?)?.toDouble(),
      inPvWindow: json['in_pv_window'] as bool?,
      error: json['error'] as String?,
    );
  }

  final String material;
  final double? egDft;
  final double? egExp;
  final double? egMl;
  final double? egMlStd;
  final double? solarScore;
  final bool? inPvWindow;
  final String? error;
}

class Top8Response {
  const Top8Response({required this.items});

  factory Top8Response.fromJson(Map<String, dynamic> json) {
    return Top8Response(
      items: [
        for (final item in (json['items'] as List? ?? const []))
          Top8Row.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
    );
  }

  final List<Top8Row> items;
}

class ModelInfo {
  const ModelInfo({
    required this.name,
    required this.metrics,
    required this.hasPickle,
    this.sizeMb,
  });

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      name: json['name'] as String? ?? '',
      metrics: Map<String, dynamic>.from(json['metrics'] as Map? ?? {}),
      hasPickle: json['has_pickle'] == true,
      sizeMb: (json['size_mb'] as num?)?.toDouble(),
    );
  }

  final String name;
  final Map<String, dynamic> metrics;
  final bool hasPickle;
  final double? sizeMb;
}

class ModelsResponse {
  const ModelsResponse({
    required this.models,
    required this.surrogateStatus,
    this.surrogateError,
  });

  factory ModelsResponse.fromJson(Map<String, dynamic> json) {
    return ModelsResponse(
      models: [
        for (final item in (json['models'] as List? ?? const []))
          ModelInfo.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      surrogateStatus: json['surrogate_status'] as String? ?? 'unknown',
      surrogateError: json['surrogate_error'] as String?,
    );
  }

  final List<ModelInfo> models;
  final String surrogateStatus;
  final String? surrogateError;
}

class ReportDoc {
  const ReportDoc({
    required this.path,
    required this.name,
    required this.group,
    required this.sizeBytes,
  });

  factory ReportDoc.fromJson(Map<String, dynamic> json) {
    return ReportDoc(
      path: json['path'] as String? ?? '',
      name: json['name'] as String? ?? '',
      group: json['group'] as String? ?? '',
      sizeBytes: (json['size_bytes'] as num? ?? 0).toInt(),
    );
  }

  final String path;
  final String name;
  final String group;
  final int sizeBytes;
}

class ReportFigure {
  const ReportFigure({
    required this.path,
    required this.name,
    required this.present,
  });

  factory ReportFigure.fromJson(Map<String, dynamic> json) {
    return ReportFigure(
      path: json['path'] as String? ?? '',
      name: json['name'] as String? ?? '',
      present: json['present'] == true,
    );
  }

  final String path;
  final String name;
  final bool present;
}

class Gallery {
  const Gallery({
    required this.name,
    required this.figures,
    required this.nDeclared,
    required this.nPresent,
    this.calculationDir,
  });

  factory Gallery.fromJson(Map<String, dynamic> json) {
    return Gallery(
      name: json['name'] as String? ?? '',
      calculationDir: json['calculation_dir'] as String?,
      figures: [
        for (final item in (json['figures'] as List? ?? const []))
          ReportFigure.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      nDeclared: (json['n_declared'] as num? ?? 0).toInt(),
      nPresent: (json['n_present'] as num? ?? 0).toInt(),
    );
  }

  final String name;
  final String? calculationDir;
  final List<ReportFigure> figures;
  final int nDeclared;
  final int nPresent;
}

class ReportsResponse {
  const ReportsResponse({required this.documents, required this.galleries});

  factory ReportsResponse.fromJson(Map<String, dynamic> json) {
    return ReportsResponse(
      documents: [
        for (final item in (json['documents'] as List? ?? const []))
          ReportDoc.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      galleries: [
        for (final item in (json['galleries'] as List? ?? const []))
          Gallery.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
    );
  }

  final List<ReportDoc> documents;
  final List<Gallery> galleries;
}

class ReportDocument {
  const ReportDocument({
    required this.path,
    required this.name,
    required this.content,
  });

  factory ReportDocument.fromJson(Map<String, dynamic> json) {
    return ReportDocument(
      path: json['path'] as String? ?? '',
      name: json['name'] as String? ?? '',
      content: json['content'] as String? ?? '',
    );
  }

  final String path;
  final String name;
  final String content;
}

class AgentHealth {
  const AgentHealth({
    required this.enabled,
    required this.provider,
    required this.ok,
    required this.baseUrl,
    required this.model,
    required this.manageService,
    required this.modelsDir,
    required this.reviveRepo,
    this.modelPresent,
    this.allowWrites,
    this.version,
    this.error,
  });

  factory AgentHealth.fromJson(Map<String, dynamic> json) {
    return AgentHealth(
      enabled: json['enabled'] == true,
      provider: json['provider'] as String? ?? '-',
      ok: json['ok'] == true,
      baseUrl: json['base_url'] as String? ?? '-',
      model: json['model'] as String? ?? '-',
      modelPresent: json['model_present'] as bool?,
      manageService: json['manage_service'] == true,
      allowWrites: json['allow_writes'] as bool?,
      modelsDir: json['models_dir'] as String? ?? '-',
      reviveRepo: json['revive_repo'] as String? ?? '-',
      version: json['version'] as String?,
      error: json['error'] as String?,
    );
  }

  final bool enabled;
  final String provider;
  final bool ok;
  final String baseUrl;
  final String model;
  final bool? modelPresent;
  final bool manageService;
  final bool? allowWrites;
  final String modelsDir;
  final String reviveRepo;
  final String? version;
  final String? error;
}

class AgentToolResult {
  const AgentToolResult({
    required this.name,
    required this.ok,
    this.arguments,
    this.data,
    this.error,
    this.statusCode,
  });

  factory AgentToolResult.fromJson(Map<String, dynamic> json) {
    return AgentToolResult(
      name: json['name'] as String? ?? '',
      ok: json['ok'] == true,
      arguments: Map<String, dynamic>.from(json['arguments'] as Map? ?? {}),
      data: json['data'],
      error: json['error'] as String?,
      statusCode: (json['status_code'] as num?)?.toInt(),
    );
  }

  final String name;
  final bool ok;
  final Map<String, dynamic>? arguments;
  final Object? data;
  final String? error;
  final int? statusCode;
}

class AgentChatResponse {
  const AgentChatResponse({
    required this.ok,
    required this.model,
    required this.message,
    required this.toolRounds,
    required this.toolResults,
    required this.proposalIds,
    this.structured,
  });

  factory AgentChatResponse.fromJson(Map<String, dynamic> json) {
    return AgentChatResponse(
      ok: json['ok'] == true,
      model: json['model'] as String? ?? '',
      message: json['message'] as String? ?? '',
      structured: json['structured'] == null
          ? null
          : Map<String, dynamic>.from(json['structured'] as Map),
      toolRounds: (json['tool_rounds'] as num? ?? 0).toInt(),
      toolResults: [
        for (final item in (json['tool_results'] as List? ?? const []))
          AgentToolResult.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      proposalIds: [
        for (final value in (json['proposal_ids'] as List? ?? const []))
          '$value',
      ],
    );
  }

  final bool ok;
  final String model;
  final String message;
  final Map<String, dynamic>? structured;
  final int toolRounds;
  final List<AgentToolResult> toolResults;
  final List<String> proposalIds;
}

class AgentProposal {
  const AgentProposal({
    required this.id,
    required this.title,
    required this.status,
    required this.executed,
    this.command,
    this.diff,
    this.rationale,
  });

  factory AgentProposal.fromJson(Map<String, dynamic> json) {
    return AgentProposal(
      id: json['id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      status: json['status'] as String? ?? '',
      executed: json['executed'] == true,
      command: json['command'] as String?,
      diff: json['diff'] as String?,
      rationale: json['rationale'] as String?,
    );
  }

  final String id;
  final String title;
  final String status;
  final bool executed;
  final String? command;
  final String? diff;
  final String? rationale;
}

/// Cola de un log de GPAW.
class LogJob {
  const LogJob({
    required this.lineas,
    required this.disponibles,
    required this.total,
    this.etiqueta,
  });

  factory LogJob.fromJson(Map<String, dynamic> json) => LogJob(
        lineas:
            ((json['lines'] as List?) ?? const []).map((e) => '$e').toList(),
        disponibles: ((json['available'] as List?) ?? const [])
            .map((e) => '$e')
            .toList(),
        total: (json['total_lines'] as num? ?? 0).toInt(),
        etiqueta: json['label'] as String?,
      );

  final List<String> lineas;

  /// Ficheros de log del trabajo: `r2scan`, `runner_stdout`, `runner_stderr`.
  final List<String> disponibles;

  final int total;
  final String? etiqueta;
}

/// Una capacidad del entorno: qué es, si funciona y cómo se arregla si no.
class SetupCapability {
  const SetupCapability({
    required this.id,
    required this.titulo,
    required this.ok,
    required this.requerido,
    required this.detalle,
    this.error,
    this.remediacion,
    this.comando,
  });

  factory SetupCapability.fromJson(Map<String, dynamic> json) {
    return SetupCapability(
      id: _textOrDash(json['id']),
      titulo: _textOrDash(json['titulo']),
      ok: json['ok'] == true,
      requerido: json['requerido'] == true,
      detalle: _mapOrEmpty(json['detalle']),
      error: _textOrNull(json['error']),
      remediacion: _textOrNull(json['remediacion']),
      comando: _textOrNull(json['comando']),
    );
  }

  final String id;
  final String titulo;
  final bool ok;
  final bool requerido;
  final Map<String, dynamic> detalle;
  final String? error;
  final String? remediacion;
  final String? comando;

  /// Versiones detectadas, para enseñar qué hay instalado sin abrir una consola.
  ///
  /// El `is! Map` no es defensivo por costumbre: una capacidad que devolviera
  /// aquí un string suelto se pintaría como un chip por carácter.
  Map<String, String> get versiones {
    final raw = detalle['versiones'];
    if (raw is! Map) return const {};
    return {
      for (final e in raw.entries) e.key.toString(): e.value.toString(),
    };
  }
}

/// Estado de la instalación en curso (o de la última que corrió).
class SetupJob {
  const SetupJob({
    required this.running,
    required this.status,
    required this.log,
    this.target,
    this.error,
  });

  factory SetupJob.fromJson(Map<String, dynamic> json) {
    return SetupJob(
      running: json['running'] == true,
      status: _textOrNull(json['status']),
      target: _textOrNull(json['target']),
      error: _textOrNull(json['error']),
      log: [
        for (final line in (json['log'] as List? ?? const [])) line.toString(),
      ],
    );
  }

  final bool running;
  final String? status;
  final String? target;
  final String? error;
  final List<String> log;

  bool get vacio => target == null && log.isEmpty;
}

/// Matriz de capacidades de la máquina donde corre el monitor.
class SetupStatus {
  const SetupStatus({
    required this.status,
    required this.ok,
    required this.plataforma,
    required this.python,
    required this.executable,
    required this.frozen,
    required this.capacidades,
    required this.job,
  });

  factory SetupStatus.fromJson(Map<String, dynamic> json) {
    return SetupStatus(
      status: _textOrDash(json['status']),
      ok: json['ok'] == true,
      plataforma: _textOrDash(json['plataforma']),
      python: _textOrDash(json['python']),
      executable: _textOrDash(json['executable']),
      frozen: json['frozen'] == true,
      capacidades: [
        for (final item in (json['capacidades'] as List? ?? const []))
          SetupCapability.fromJson(Map<String, dynamic>.from(item as Map)),
      ],
      job: SetupJob.fromJson(_mapOrEmpty(json['job'])),
    );
  }

  final String status;
  final bool ok;
  final String plataforma;
  final String python;
  final String executable;
  final bool frozen;
  final List<SetupCapability> capacidades;
  final SetupJob job;
}
