import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/providers.dart';
import '../models/models.dart';
import '../repositories/repositories.dart';
import '../utils/format.dart';
import '../widgets/async_panel.dart';
import '../widgets/status_chip.dart';

class DashboardView extends ConsumerWidget {
  const DashboardView({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final health = ref.watch(healthProvider);
    final summary = ref.watch(summaryProvider);
    final system = ref.watch(systemMetricsProvider);
    final history = ref.watch(systemHistoryProvider);
    final batches = ref.watch(batchesProvider);
    final activeJobs = ref.watch(activeJobsProvider);
    final events = ref.watch(wsEventLogProvider);

    return RefreshIndicator(
      onRefresh: () async {
        ref.invalidate(healthProvider);
        ref.invalidate(summaryProvider);
        ref.invalidate(systemMetricsProvider);
        ref.invalidate(systemHistoryProvider);
        ref.invalidate(batchesProvider);
        ref.invalidate(activeJobsProvider);
      },
      child: ListView(
        children: [
          LayoutBuilder(
            builder: (context, constraints) {
              final columns = constraints.maxWidth >= 1120
                  ? 3
                  : constraints.maxWidth >= 720
                      ? 2
                      : 1;
              final itemWidth =
                  (constraints.maxWidth - (columns - 1) * 12) / columns;
              final topCardHeight = columns == 3 ? 286.0 : null;

              return Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  SizedBox(
                    width: itemWidth,
                    height: topCardHeight,
                    child: Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: AsyncPanel(
                          value: health,
                          builder: (data) => Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              _Header(
                                  title: 'Salud',
                                  value: data.ok ? 'OK' : 'Atencion'),
                              const SizedBox(height: 12),
                              _Line('Versión', data.version),
                              _Line('Jobs rastreados', '${data.nJobsTracked}'),
                              _Line(
                                  'Poll',
                                  data.lastPollAgeSec == null
                                      ? '-'
                                      : '${fmtNumber(data.lastPollAgeSec, 1)} s'),
                              _Line('WebSocket', '${data.wsClients} clientes'),
                              _Line('Runs', data.runsDir),
                              _Line(
                                  'Volumen',
                                  switch (data.runsState) {
                                    'montado' => 'montado',
                                    'sin_estrenar' => 'sin estrenar',
                                    _ => 'DESMONTADO',
                                  },
                                  alerta: data.runsState == 'desmontado'),
                              _Line(
                                  'Auto-avance',
                                  data.autoAdvance
                                      ? 'activo - puede mutar el pipeline'
                                      : 'solo observa',
                                  alerta: data.autoAdvance),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    width: itemWidth,
                    height: topCardHeight,
                    child: Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: AsyncPanel(
                          value: summary,
                          builder: (data) => Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              _Header(title: 'Jobs', value: '${data.total}'),
                              const SizedBox(height: 12),
                              _JobsMetricGrid(
                                metrics: [
                                  _JobsMetric('calculando', data.running),
                                  _JobsMetric('en cola', data.pending),
                                  _JobsMetric('convergidos', data.converged),
                                  _JobsMetric('fallidos', data.failed),
                                  _JobsMetric('estancados',
                                      data.stalled + data.oscillating),
                                  _JobsMetric('duplicados', data.skippedDuplicate),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    width: itemWidth,
                    height: topCardHeight,
                    child: Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: AsyncPanel(
                          value: system,
                          builder: (data) => Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const _Header(title: 'Sistema', value: ''),
                              const SizedBox(height: 12),
                              _Line(
                                  'CPU', '${fmtNumber(data.cpuPercent, 1)} %'),
                              _Line('RAM',
                                  '${fmtNumber(data.ramUsedGb, 1)} / ${fmtNumber(data.ramTotalGb, 1)} GB'),
                              _Line('Temp CPU',
                                  '${fmtNumber(data.coreTempMax, 1)} C'),
                              if (data.nvmeTemp != null)
                                _Line(
                                    'NVMe', '${fmtNumber(data.nvmeTemp, 1)} C'),
                              _Line(
                                  'GPU',
                                  data.gpuTemps
                                      .map((v) => '${fmtNumber(v, 0)} C')
                                      .join(' · ')),
                              if (data.cpuPerCore.isNotEmpty) ...[
                                const SizedBox(height: 8),
                                _CoreGrid(values: data.cpuPerCore),
                              ],
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              );
            },
          ),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: AsyncPanel(
                value: history,
                builder: (data) => Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const _Header(title: 'Hardware', value: ''),
                    const SizedBox(height: 12),
                    SizedBox(
                      height: 96,
                      child: Row(
                        children: [
                          _SparkMetric(
                            label: 'CPU',
                            value:
                                data.samples.map((s) => s.cpuPercent).toList(),
                            max: 100,
                          ),
                          const SizedBox(width: 12),
                          _SparkMetric(
                            label: 'RAM',
                            value:
                                data.samples.map((s) => s.ramPercent).toList(),
                            max: 100,
                          ),
                          const SizedBox(width: 12),
                          _SparkMetric(
                            label: 'CPU temp',
                            value:
                                data.samples.map((s) => s.coreTempMax).toList(),
                            max: 100,
                          ),
                          const SizedBox(width: 12),
                          _SparkMetric(
                            label: 'GPU temp',
                            value: data.samples
                                .map((s) => s.gpuTempMax ?? 0)
                                .toList(),
                            max: 100,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: AsyncPanel(
                value: batches,
                builder: (items) => Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const _Header(title: 'Lotes', value: ''),
                    const SizedBox(height: 8),
                    LayoutBuilder(
                      builder: (context, constraints) {
                        return SingleChildScrollView(
                          scrollDirection: Axis.horizontal,
                          child: ConstrainedBox(
                            constraints:
                                BoxConstraints(minWidth: constraints.maxWidth),
                            child: DataTable(
                              columns: const [
                                DataColumn(label: Text('Lote')),
                                DataColumn(label: Text('Estado')),
                                DataColumn(label: Text('Ruta')),
                              ],
                              rows: [
                                for (final batch in items.take(12))
                                  DataRow(cells: [
                                    DataCell(Text(batch.batchId
                                        .toString()
                                        .padLeft(3, '0'))),
                                    DataCell(Text(
                                      batch.counts.entries
                                          .map((e) => '${e.key}:${e.value}')
                                          .join('  '),
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                    )),
                                    DataCell(ConstrainedBox(
                                      constraints: BoxConstraints(
                                          maxWidth:
                                              constraints.maxWidth * 0.52),
                                      child: Text(batch.path,
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis),
                                    )),
                                  ]),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          LayoutBuilder(
            builder: (context, constraints) {
              final activeCard = _ActiveJobsCard(value: activeJobs);
              final eventsCard = _EventsCard(value: events);
              if (constraints.maxWidth < 860) {
                return Column(
                  children: [
                    activeCard,
                    const SizedBox(height: 16),
                    eventsCard,
                  ],
                );
              }
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(flex: 2, child: activeCard),
                  const SizedBox(width: 16),
                  Expanded(child: eventsCard),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}

class _ActiveJobsCard extends StatelessWidget {
  const _ActiveJobsCard({required this.value});

  final AsyncValue<JobPage> value;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: AsyncPanel(
          value: value,
          builder: (page) => Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _Header(title: 'Jobs activos', value: '${page.total}'),
              const SizedBox(height: 8),
              if (page.items.isEmpty)
                Text('Ningún trabajo activo.',
                    style: Theme.of(context).textTheme.bodySmall)
              else
                for (final job in page.items)
                  ListTile(
                    dense: true,
                    contentPadding: EdgeInsets.zero,
                    leading: StatusChip(job.status),
                    title: Text(fmtFormula(job.formula),
                        overflow: TextOverflow.ellipsis),
                    subtitle: Text(job.jobId, overflow: TextOverflow.ellipsis),
                    trailing: Text(fmtMinutes(job.elapsedMin)),
                  ),
            ],
          ),
        ),
      ),
    );
  }
}

class _EventsCard extends StatelessWidget {
  const _EventsCard({required this.value});

  final AsyncValue<List<String>> value;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: AsyncPanel(
          value: value,
          builder: (items) => Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _Header(title: 'Eventos', value: '${items.length}'),
              const SizedBox(height: 8),
              if (items.isEmpty)
                Text('Sin eventos recientes.',
                    style: Theme.of(context).textTheme.bodySmall)
              else
                for (final event in items.take(10)) _EventLine(raw: event),
            ],
          ),
        ),
      ),
    );
  }
}

class _CoreGrid extends StatelessWidget {
  const _CoreGrid({required this.values});

  final List<double> values;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 3,
      runSpacing: 3,
      children: [
        for (var i = 0; i < values.length; i++)
          Tooltip(
            message: 'core $i: ${fmtNumber(values[i], 0)}%',
            child: Container(
              width: 14,
              height: 14,
              decoration: BoxDecoration(
                color: Color.lerp(const Color(0xff111827),
                    const Color(0xff60a5fa), values[i].clamp(0, 100) / 100),
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),
      ],
    );
  }
}

class _SparkMetric extends StatelessWidget {
  const _SparkMetric({
    required this.label,
    required this.value,
    required this.max,
  });

  final String label;
  final List<double> value;
  final double max;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 6),
          Expanded(
            child: CustomPaint(
              painter: _SparkPainter(
                  values: value,
                  max: max,
                  color: Theme.of(context).colorScheme.primary),
              child: const SizedBox.expand(),
            ),
          ),
        ],
      ),
    );
  }
}

class _SparkPainter extends CustomPainter {
  const _SparkPainter({
    required this.values,
    required this.max,
    required this.color,
  });

  final List<double> values;
  final double max;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final grid = Paint()
      ..color = const Color(0xff263244)
      ..strokeWidth = 1;
    canvas.drawLine(
        Offset(0, size.height), Offset(size.width, size.height), grid);
    if (values.length < 2) return;
    final path = Path();
    for (var i = 0; i < values.length; i++) {
      final x = size.width * i / (values.length - 1);
      final y = size.height - (values[i].clamp(0, max) / max) * size.height;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(covariant _SparkPainter oldDelegate) =>
      oldDelegate.values != values ||
      oldDelegate.max != max ||
      oldDelegate.color != color;
}

class _EventLine extends StatelessWidget {
  const _EventLine({required this.raw});

  final String raw;

  @override
  Widget build(BuildContext context) {
    Map<String, dynamic> data;
    try {
      data = Map<String, dynamic>.from(jsonDecode(raw) as Map);
    } catch (_) {
      data = {'event': 'event', 'job_id': raw};
    }
    final timestamp = '${data['timestamp'] ?? ''}';
    final time = timestamp.length >= 19 ? timestamp.substring(11, 19) : '';
    final event = '${data['event'] ?? '-'}';
    final jobId = '${data['job_id'] ?? ''}';
    final payload = Map<String, dynamic>.from(data['data'] as Map? ?? {});
    final formula = '${payload['formula'] ?? jobId}';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          SizedBox(
              width: 58,
              child: Text(time, style: Theme.of(context).textTheme.bodySmall)),
          SizedBox(
              width: 86, child: Text(event, overflow: TextOverflow.ellipsis)),
          Expanded(
              child:
                  Text(fmtFormula(formula), overflow: TextOverflow.ellipsis)),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.title, required this.value});

  final String title;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(title.toUpperCase(),
            style: Theme.of(context).textTheme.labelSmall),
        const Spacer(),
        if (value.isNotEmpty)
          Text(value, style: Theme.of(context).textTheme.titleLarge),
      ],
    );
  }
}

class _Line extends StatelessWidget {
  const _Line(this.label, this.value, {this.alerta = false});

  final String label;
  final String value;

  /// Resalta el valor cuando merece atencion: volumen desmontado, auto-avance
  /// encendido. Sin esto los dos campos quedaban invisibles y un panel vacio no
  /// se distinguia de un disco sin montar.
  final bool alerta;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
              width: 116,
              child: Text(label, style: Theme.of(context).textTheme.bodySmall)),
          Expanded(
            child: Text(
              value,
              style: alerta
                  ? TextStyle(
                      color: Theme.of(context).colorScheme.error,
                      fontWeight: FontWeight.w600)
                  : null,
            ),
          ),
        ],
      ),
    );
  }
}

class _JobsMetric {
  const _JobsMetric(this.label, this.value);

  final String label;
  final int value;
}

class _JobsMetricGrid extends StatelessWidget {
  const _JobsMetricGrid({required this.metrics});

  final List<_JobsMetric> metrics;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = constraints.maxWidth >= 300 ? 3 : 2;
        final tileWidth = (constraints.maxWidth - (columns - 1) * 8) / columns;
        return Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final metric in metrics)
              SizedBox(
                width: tileWidth,
                child: _Metric(metric.label, metric.value),
              ),
          ],
        );
      },
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric(this.label, this.value);

  final String label;
  final int value;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 56,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xff0b1220),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xff1f2937)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.bodySmall,
          ),
          Text('$value', style: Theme.of(context).textTheme.titleLarge),
        ],
      ),
    );
  }
}
