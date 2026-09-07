import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/providers.dart';
import '../repositories/repositories.dart';

/// Arranque de un clic del protocolo autónomo.
///
/// Mide la máquina, fija el reparto de trabajos, comprueba que estén las
/// piezas y lanza el bucle. Muestra el reparto y la calidad actual del
/// surrogate ANTES de arrancar: el protocolo gasta días de DFT guiado por ese
/// modelo, así que si apenas mejora sobre predecir la media conviene saberlo
/// antes de encenderlo, no después.
class QuickStartCard extends ConsumerStatefulWidget {
  const QuickStartCard({super.key});

  @override
  ConsumerState<QuickStartCard> createState() => _QuickStartCardState();
}

class _QuickStartCardState extends ConsumerState<QuickStartCard> {
  bool _lanzando = false;
  Map<String, dynamic>? _resultado;
  String? _error;

  Future<void> _arrancar() async {
    setState(() {
      _lanzando = true;
      _error = null;
    });
    try {
      final api = ref.read(apiClientProvider);
      final r = await api.postMap('/api/quickstart', body: const {});
      if (!mounted) return;
      setState(() => _resultado = r);
      ref.invalidate(quickStartStatusProvider);
      ref.invalidate(hardwareProbeProvider);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _lanzando = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final probe = ref.watch(hardwareProbeProvider);
    final estado = ref.watch(quickStartStatusProvider);
    final theme = Theme.of(context);

    final protocolo = estado.asData?.value['protocolo'] as Map<String, dynamic>?;
    final estadoProtocolo =
        (protocolo?['state'] as Map<String, dynamic>?)?['status'] as String?;
    final corriendo = estadoProtocolo != null &&
        estadoProtocolo != 'not_initialized' &&
        estadoProtocolo != 'done' &&
        estadoProtocolo != 'error';

    final surrogate = estado.asData?.value['surrogate'] as Map<String, dynamic>?;
    final mejora = surrogate?['mejora_sobre_media_pct'] as num?;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Icon(Icons.rocket_launch_outlined,
                    size: 20, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text('Arranque rápido',
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.w600)),
                const Spacer(),
                if (corriendo)
                  Chip(
                    visualDensity: VisualDensity.compact,
                    label: Text('protocolo $estadoProtocolo'),
                  ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Mide la máquina, ajusta cuántos cálculos correr en paralelo y '
              'lanza el protocolo autónomo completo.',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            probe.when(
              loading: () => const Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: LinearProgressIndicator(),
              ),
              error: (e, _) => Text('No se pudo medir la máquina: $e',
                  style: TextStyle(color: theme.colorScheme.error)),
              data: (d) => _Reparto(datos: d),
            ),
            if (mejora != null) ...[
              const SizedBox(height: 10),
              _AvisoSurrogate(mejora: mejora.toDouble(), datos: surrogate!),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: _lanzando || corriendo ? null : _arrancar,
                  icon: _lanzando
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.play_arrow),
                  label: Text(corriendo
                      ? 'Protocolo en marcha'
                      : _lanzando
                          ? 'Arrancando…'
                          : 'Arrancar protocolo'),
                ),
                const SizedBox(width: 8),
                TextButton(
                  onPressed: _lanzando
                      ? null
                      : () => ref.invalidate(hardwareProbeProvider),
                  child: const Text('Volver a medir'),
                ),
              ],
            ),
            if (_error != null) ...[
              const SizedBox(height: 8),
              Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
            ],
            if (_resultado != null) ...[
              const SizedBox(height: 8),
              _Pasos(resultado: _resultado!),
            ],
          ],
        ),
      ),
    );
  }
}

class _Reparto extends StatelessWidget {
  const _Reparto({required this.datos});

  final Map<String, dynamic> datos;

  @override
  Widget build(BuildContext context) {
    final m = datos['maquina'] as Map<String, dynamic>? ?? {};
    final r = datos['reparto'] as Map<String, dynamic>? ?? {};
    final theme = Theme.of(context);
    final bench = m['benchmark'] as Map<String, dynamic>? ?? {};

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 6,
          children: [
            _Pastilla(
                icono: Icons.memory,
                texto: '${m['nucleos_fisicos']} núcleos'
                    '${m['smt'] == true ? ' (${m['nucleos_logicos']} hilos)' : ''}'),
            if (m['frecuencia_mhz'] != null)
              _Pastilla(
                  icono: Icons.speed,
                  texto:
                      '${((m['frecuencia_mhz'] as num) / 1000).toStringAsFixed(2)} GHz'),
            _Pastilla(
                icono: Icons.storage,
                texto:
                    '${m['ram_disponible_gb']} / ${m['ram_total_gb']} GB libres'),
            if (bench['disponible'] == true)
              _Pastilla(icono: Icons.bolt, texto: '${bench['gflops']} GFLOP/s'),
          ],
        ),
        const SizedBox(height: 10),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color:
                theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${r['runner_slots']} cálculos en paralelo × '
                '${r['runner_cores']} núcleos cada uno',
                style: theme.textTheme.bodyMedium
                    ?.copyWith(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 2),
              Text(
                'Limitado por ${r['limitado_por'] == 'ram' ? 'la memoria' : 'los núcleos'}'
                ' · ~${r['ram_comprometida_gb']} GB comprometidos',
                style: theme.textTheme.bodySmall,
              ),
              if (r['aviso'] != null) ...[
                const SizedBox(height: 4),
                Text(r['aviso'] as String,
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: theme.colorScheme.error)),
              ],
              const SizedBox(height: 4),
              // Que no se confunda con la calibración real: esta estimación
              // dimensiona a partir de recursos, no mide t/iteración con GPAW.
              Text(
                'Estimado a partir de los recursos de la máquina. Para medir el '
                'óptimo real, usa el barrido de calibración.',
                style: theme.textTheme.bodySmall
                    ?.copyWith(fontStyle: FontStyle.italic),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _AvisoSurrogate extends StatelessWidget {
  const _AvisoSurrogate({required this.mejora, required this.datos});

  final double mejora;
  final Map<String, dynamic> datos;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    // Por debajo de este margen el surrogate no separa candidatos mejor que
    // predecir la media, así que el DFT se repartiría casi al azar.
    final flojo = mejora < 25.0;
    if (!flojo) {
      return Text(
        'Surrogate: ${mejora.toStringAsFixed(0)} % mejor que predecir la media '
        '(${datos['n_samples']} muestras).',
        style: theme.textTheme.bodySmall,
      );
    }
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: theme.colorScheme.errorContainer.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.warning_amber_rounded,
              size: 18, color: theme.colorScheme.error),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'El surrogate solo acierta un ${mejora.toStringAsFixed(0)} % mejor '
              'que predecir la media (${datos['n_samples']} muestras, '
              'cv_mae ${datos['cv_mae_eV']} eV). Con tan poco margen el '
              'protocolo elegirá candidatos casi al azar: el DFT se gasta igual, '
              'pero guía poco. Es normal en las primeras rondas.',
              style: theme.textTheme.bodySmall,
            ),
          ),
        ],
      ),
    );
  }
}

class _Pasos extends StatelessWidget {
  const _Pasos({required this.resultado});

  final Map<String, dynamic> resultado;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pasos = (resultado['pasos'] as List?) ?? const [];
    final ok = resultado['ok'] == true;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          ok
              ? 'Protocolo arrancado en ${resultado['segundos']} s'
              : 'No arrancó: ${resultado['motivo']}',
          style: theme.textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.w600,
            color: ok ? null : theme.colorScheme.error,
          ),
        ),
        const SizedBox(height: 4),
        for (final p in pasos.cast<Map<String, dynamic>>())
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 1),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  p['ok'] == true
                      ? Icons.check_circle_outline
                      : Icons.error_outline,
                  size: 15,
                  color: p['ok'] == true
                      ? theme.colorScheme.primary
                      : theme.colorScheme.error,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    p['error'] != null
                        ? '${p['paso']}: ${p['error']}'
                        : '${p['paso']}',
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
        ..._faltantes(context, pasos),
      ],
    );
  }

  /// Lo que falta, con su remediación: un "no arrancó" sin decir qué instalar
  /// deja al usuario en el mismo sitio que antes de pulsar.
  List<Widget> _faltantes(BuildContext context, List pasos) {
    final theme = Theme.of(context);
    for (final p in pasos.cast<Map<String, dynamic>>()) {
      if (p['paso'] != 'prerequisitos') continue;
      final faltan = ((p['detalle'] as Map<String, dynamic>?)?['faltantes']
              as List?) ??
          const [];
      if (faltan.isEmpty) return const [];
      return [
        const SizedBox(height: 6),
        for (final f in faltan.cast<Map<String, dynamic>>())
          Padding(
            padding: const EdgeInsets.only(left: 21, top: 2),
            child: Text(
              '· ${f['titulo']}: ${f['remediacion'] ?? f['error'] ?? 'no disponible'}',
              style: theme.textTheme.bodySmall
                  ?.copyWith(color: theme.colorScheme.error),
            ),
          ),
        // El arranque rápido no instala nada por su cuenta: el entorno DFT son
        // ~2.5 GB de descarga, y encadenarlo a un clic sería descargarlos sin
        // avisar. Se manda a Entorno, que sí pide confirmación con el plan.
        Padding(
          padding: const EdgeInsets.only(left: 15, top: 6),
          child: Text(
            'Instálalo desde la pestaña Entorno.',
            style: theme.textTheme.bodySmall
                ?.copyWith(fontStyle: FontStyle.italic),
          ),
        ),
      ];
    }
    return const [];
  }
}

class _Pastilla extends StatelessWidget {
  const _Pastilla({required this.icono, required this.texto});

  final IconData icono;
  final String texto;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        border: Border.all(color: theme.dividerColor),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icono, size: 14),
          const SizedBox(width: 5),
          Text(texto, style: theme.textTheme.bodySmall),
        ],
      ),
    );
  }
}
