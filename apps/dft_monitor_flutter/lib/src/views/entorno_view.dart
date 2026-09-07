import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/errors.dart';
import '../models/models.dart';
import '../repositories/repositories.dart';
import '../widgets/async_panel.dart';
import '../widgets/confirm_action.dart';

/// Wizard de entorno: qué runtime funciona, qué falta y cómo instalarlo.
///
/// PEROVOWL no corre en un solo intérprete —el monitor, GPAW y el MLFF viven
/// en entornos distintos— y hasta ahora que faltara uno solo se descubría a
/// mitad de una ronda, como un `ModuleNotFoundError` en el log. Esta pantalla
/// adelanta esa comprobación y la convierte en un botón.
class EntornoView extends ConsumerStatefulWidget {
  const EntornoView({super.key});

  @override
  ConsumerState<EntornoView> createState() => _EntornoViewState();
}

class _EntornoViewState extends ConsumerState<EntornoView> {
  void _recargar() {
    ref.invalidate(setupStatusProvider(true));
    ref.invalidate(setupStatusProvider(false));
    ref.invalidate(setupJobProvider);
  }

  @override
  Widget build(BuildContext context) {
    // Las dos consultas viven a la vez. La rápida omite la sonda MLFF, que
    // lanza un proceso (y en Windows, una distro de WSL); la completa la
    // incluye. Se pinta la que haya, prefiriendo la completa: con una sola
    // consulta que cambiara de `fast` a no-`fast`, la clave del provider
    // cambiaría y la pantalla volvería al spinner, que es justo lo que este
    // arreglo evita.
    final rapido = ref.watch(setupStatusProvider(true));
    final completo = ref.watch(setupStatusProvider(false));
    final datos = completo.valueOrNull ?? rapido.valueOrNull;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text('Entorno', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(width: 12),
            if (datos != null) _ChipEstado(status: datos.status),
            if (datos != null && completo.isLoading) ...[
              const SizedBox(width: 8),
              Text('sondeando MLFF…',
                  style: Theme.of(context).textTheme.labelSmall),
            ],
            const Spacer(),
            OutlinedButton.icon(
              onPressed: _recargar,
              icon: const Icon(Icons.refresh),
              label: const Text('Recomprobar'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Expanded(
          child: datos != null
              ? _Contenido(data: datos, onCambio: _recargar)
              // Sin datos de ninguna de las dos: se enseña el error de la
              // rápida, que es la que ya debería haber respondido.
              : AsyncPanel<SetupStatus>(
                  value: rapido,
                  builder: (data) =>
                      _Contenido(data: data, onCambio: _recargar),
                ),
        ),
      ],
    );
  }
}

class _ChipEstado extends StatelessWidget {
  const _ChipEstado({required this.status});

  final String status;

  @override
  Widget build(BuildContext context) {
    final esquema = Theme.of(context).colorScheme;
    final (color, texto) = switch (status) {
      'ok' => (esquema.primary, 'Todo listo'),
      'degradado' => (Colors.orange, 'Funciona con limitaciones'),
      _ => (esquema.error, 'Falta algo esencial'),
    };
    return Chip(
      visualDensity: VisualDensity.compact,
      side: BorderSide(color: color),
      backgroundColor: color.withValues(alpha: 0.12),
      label: Text(texto, style: TextStyle(color: color, fontSize: 12)),
    );
  }
}

class _Contenido extends ConsumerWidget {
  const _Contenido({required this.data, required this.onCambio});

  final SetupStatus data;
  final VoidCallback onCambio;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final job = ref.watch(setupJobProvider);
    final corriendo = job.valueOrNull?.running ?? false;

    return ListView(
      children: [
        if (data.frozen)
          const Card(
            child: ListTile(
              leading: Icon(Icons.lock_outline),
              title: Text('Monitor congelado'),
              subtitle: Text(
                'Este monitor es un binario empaquetado: no tiene dónde instalar '
                'paquetes de Python. Las dependencias que corren fuera (WSL) sí '
                'se pueden instalar desde aquí.',
              ),
            ),
          ),
        for (final cap in data.capacidades)
          _TarjetaCapacidad(
            cap: cap,
            bloqueado: corriendo,
            onCambio: onCambio,
          ),
        const SizedBox(height: 12),
        if (!(job.valueOrNull?.vacio ?? true)) _PanelInstalacion(job: job),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('INTÉRPRETE DEL MONITOR',
                    style: Theme.of(context).textTheme.labelSmall),
                const SizedBox(height: 6),
                SelectableText('Python ${data.python} · ${data.plataforma}',
                    style: const TextStyle(fontSize: 12)),
                SelectableText(data.executable,
                    style: const TextStyle(fontSize: 11, fontFamily: 'monospace')),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _TarjetaCapacidad extends ConsumerWidget {
  const _TarjetaCapacidad({
    required this.cap,
    required this.bloqueado,
    required this.onCambio,
  });

  final SetupCapability cap;
  final bool bloqueado;
  final VoidCallback onCambio;

  /// El runtime DFT y el MLFF se instalan desde la GUI; los demás grupos van
  /// al intérprete del propio monitor, que no puede reinstalarse a sí mismo
  /// mientras sirve.
  ///
  /// El de DFT no instala WSL: eso exige administrador y reiniciar. Si falta,
  /// el plan que devuelve el servidor trae el comando exacto y ningún paso, y
  /// el diálogo lo enseña tal cual.
  bool get _instalable => cap.id == 'mlff' || cap.id == 'dft';

  String get _nombreEntorno => cap.id == 'dft' ? 'DFT (GPAW en WSL)' : 'MLFF';

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final esquema = Theme.of(context).colorScheme;
    final color = cap.ok
        ? esquema.primary
        : (cap.requerido ? esquema.error : Colors.orange);
    final versiones = cap.versiones;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(cap.ok ? Icons.check_circle : Icons.error_outline,
                    color: color, size: 20),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(cap.titulo,
                      style: Theme.of(context).textTheme.titleSmall),
                ),
                if (!cap.requerido)
                  Text('opcional',
                      style: Theme.of(context).textTheme.labelSmall),
                if (!cap.ok && _instalable) ...[
                  const SizedBox(width: 8),
                  FilledButton.icon(
                    onPressed: bloqueado
                        ? null
                        : () => _instalar(context, ref, recrear: false),
                    icon: const Icon(Icons.download, size: 16),
                    label: const Text('Instalar'),
                  ),
                ],
                if (cap.ok && _instalable) ...[
                  const SizedBox(width: 8),
                  TextButton(
                    onPressed: bloqueado
                        ? null
                        : () => _instalar(context, ref, recrear: true),
                    child: const Text('Reinstalar'),
                  ),
                ],
              ],
            ),
            if (versiones.isNotEmpty) ...[
              const SizedBox(height: 6),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                children: [
                  for (final e in versiones.entries)
                    Chip(
                      visualDensity: VisualDensity.compact,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      label: Text('${e.key} ${e.value}',
                          style: const TextStyle(fontSize: 11)),
                    ),
                ],
              ),
            ],
            if (!cap.ok) ...[
              const SizedBox(height: 6),
              if (cap.error != null)
                SelectableText(cap.error!,
                    style: TextStyle(fontSize: 12, color: esquema.error)),
              if (cap.remediacion != null && cap.remediacion!.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(cap.remediacion!,
                      style: const TextStyle(fontSize: 12)),
                ),
              if (cap.comando != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: SelectableText('\$ ${cap.comando}',
                      style: const TextStyle(
                          fontSize: 11, fontFamily: 'monospace')),
                ),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _instalar(BuildContext context, WidgetRef ref,
      {required bool recrear}) async {
    final plan = await _pedirPlan(context, ref);
    if (plan == null || !context.mounted) return;

    final ok = await confirmAction(
      context,
      title: recrear
          ? 'Reinstalar el entorno $_nombreEntorno'
          : 'Instalar el entorno $_nombreEntorno',
      message: plan,
      confirmLabel: 'Instalar',
    );
    if (!ok || !context.mounted) return;

    try {
      await ref.read(setupActionsProvider).install(cap.id, recreate: recrear);
      onCambio();
    } catch (error) {
      if (context.mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text(mensajeDeError(error))));
      }
    }
  }

  Future<String?> _pedirPlan(BuildContext context, WidgetRef ref) async {
    try {
      final plan = await ref.read(setupActionsProvider).plan(cap.id);
      final pasos = (plan['steps'] as List? ?? const [])
          .map((s) => '· ${(s as Map)['descripcion']}')
          .join('\n');
      final notas = (plan['notas'] as List? ?? const [])
          .map((n) => '· $n')
          .join('\n');
      // Un plan sin pasos no es un error: es «esto no se puede hacer desde
      // aquí», con el comando que sí lo hace. Instalar WSL es ese caso — pide
      // administrador y reiniciar, así que la app guía en vez de fingir.
      if (pasos.isEmpty) {
        return notas.isEmpty
            ? 'No hay nada que instalar.'
            : 'Esto no se puede instalar desde la app:\n\n$notas';
      }
      return [
        'Se van a ejecutar:\n$pasos',
        if (notas.isNotEmpty) '\n$notas',
      ].join('\n');
    } catch (error) {
      if (context.mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text(mensajeDeError(error))));
      }
      return null;
    }
  }
}

class _PanelInstalacion extends StatelessWidget {
  const _PanelInstalacion({required this.job});

  final AsyncValue<SetupJob> job;

  @override
  Widget build(BuildContext context) {
    return AsyncPanel<SetupJob>(
      value: job,
      height: 80,
      builder: (data) {
        final esquema = Theme.of(context).colorScheme;
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    if (data.running)
                      const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    else
                      Icon(
                        data.status == 'ok' ? Icons.check_circle : Icons.error_outline,
                        size: 18,
                        color: data.status == 'ok' ? esquema.primary : esquema.error,
                      ),
                    const SizedBox(width: 8),
                    Text(
                      data.running
                          ? 'Instalando ${data.target ?? ''}…'
                          : 'Instalación de ${data.target ?? ''}: ${data.status ?? ''}',
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                  ],
                ),
                if (data.error != null) ...[
                  const SizedBox(height: 6),
                  SelectableText(data.error!,
                      style: TextStyle(fontSize: 12, color: esquema.error)),
                ],
                const SizedBox(height: 8),
                // Sólo la cola: pip escribe miles de líneas y lo único que
                // importa mientras corre es en qué va y si falló.
                Container(
                  height: 220,
                  width: double.infinity,
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: esquema.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: SingleChildScrollView(
                    reverse: true,
                    child: SelectableText(
                      data.log.isEmpty ? '(sin salida todavía)' : data.log.join('\n'),
                      style: const TextStyle(
                          fontFamily: 'monospace', fontSize: 11),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
