"""Measured tables, publication plots and an IEEE-compatible LaTeX results section."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--campaign-dir', required=True)
a = p.parse_args()
campaign = Path(a.campaign_dir)
plan = json.loads((campaign/'plan.json').read_text())
output = campaign/'report'
output.mkdir(exist_ok=False)
figures = output/'figures'; figures.mkdir()
reuse = json.loads((campaign/'reused-short-comparison.json').read_text())


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def csv_read(path):
    if not path.exists(): return []
    with path.open() as file: return list(csv.DictReader(file))


def csv_write(name, rows):
    if not rows: return
    with (output/name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


initial = read(campaign/'initial/summary.json')
initial_reuse = read(campaign/'initial/reuse.json')
initial_eval = initial['evaluations'][0] if initial and initial.get('evaluations') else None
measurements, evals, intervals = [], [], []
states = {}
for method in ['zero','lsp']:
    summary = read(campaign/method/'summary.json')
    if not summary: continue
    states[method] = summary
    command = read(campaign/(method+'-command.json'))
    peaks = summary.get('memory_peaks', {})
    last = summary.get('evaluations', [])[-1] if summary.get('evaluations') else None
    measurements.append(dict(method=method, status=summary['status'], completed_steps=summary['completed_steps'],
        python_total_s=summary.get('real_total_seconds'), external_stage_s=command['real_stage_seconds'] if command else None,
        training_s=summary.get('training_seconds', sum(x['seconds'] for x in summary['training_intervals']) if summary['training_intervals'] else None),
        seconds_per_iteration=summary.get('seconds_per_iteration'), iterations_per_second=summary.get('iterations_per_second'),
        calibration_s=sum(x['seconds'] for x in summary['calibrations']),
        evaluation_s=sum(x['seconds'] for x in summary['evaluations']),
        weight_save_s=summary.get('weight_save_seconds'),
        final_evaluated_step=last['step'] if last else None, final_eval_loss=last['mean_batch_loss'] if last else None,
        final_eval_perplexity=last['perplexity'] if last else None,
        peak_rss_gib=peaks['rss_bytes']/2**30 if 'rss_bytes' in peaks else None,
        peak_gpu_allocated_gib=peaks['gpu_max_allocated_bytes']/2**30 if 'gpu_max_allocated_bytes' in peaks else None,
        peak_gpu_reserved_gib=peaks['gpu_max_reserved_bytes']/2**30 if 'gpu_max_reserved_bytes' in peaks else None))
    if initial_eval:
        evals.append(dict(method=method, step=0, loss=initial_eval['mean_batch_loss'],
            perplexity=initial_eval['perplexity'], instrumented_seconds=0.,
            evaluation_seconds=initial_eval['seconds'], examples=initial_eval['examples'], shared_initial=True))
    for e in summary.get('evaluations', []):
        evals.append(dict(method=method, step=e['step'], loss=e['mean_batch_loss'], perplexity=e['perplexity'],
            instrumented_seconds=e['elapsed_seconds'], evaluation_seconds=e['seconds'],
            examples=e['examples'], shared_initial=False))
    intervals.extend(dict(method=method, **x) for x in summary['training_intervals'])
csv_write('main_measurements.csv', measurements)
csv_write('full_evaluations.csv', evals)
csv_write('training_intervals.csv', intervals)
micro = []
for run in reuse['runs']:
    s = run['summary']
    if s['kind'] != 'performance': continue
    w = s['performance_window']
    micro.append(dict(run_id=s['run_id'], method=s['method'], iterations=w['iterations'], warmup=5,
        seconds=w['seconds'], seconds_per_iteration=w['seconds_per_iteration'],
        iterations_per_second=w['iterations_per_second'], reused=True,
        code_commit=plan['reuse']['performance_code_commit'], version_scope='historical_before_patch_0003'))
csv_write('reused_microbenchmarks.csv', micro)

plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'legend.fontsize':7,
                     'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
colors = {'zero':'#355C91','lsp':'#C75B39'}
labels = {'zero':'Zero-Offload','lsp':'LSP-Offload'}
generated = []


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(figures/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(figures/(name+'.png'), dpi=220, bbox_inches='tight')
    plt.close(fig); generated.append(name)


if evals:
    for field, axis_label in [('perplexity','Perplexity de evaluación'),('loss','Pérdida de evaluación (media por lote)')]:
        fig, ax = plt.subplots(figsize=(3.5,2.6))
        for method in ['zero','lsp']:
            rows = [x for x in evals if x['method']==method]
            if rows: ax.plot([x['step'] for x in rows],[x[field] for x in rows],marker='o',markersize=3,
                             label=labels[method],color=colors[method])
        ax.set_xlabel('Pasos de entrenamiento'); ax.set_ylabel(axis_label)
        ax.grid(alpha=.2); ax.legend(); finish(fig, 'evaluation_'+field+'_steps')
    fig, ax = plt.subplots(figsize=(3.5,2.6))
    for method in ['zero','lsp']:
        rows = [x for x in evals if x['method']==method]
        if rows: ax.plot([x['instrumented_seconds']/3600 for x in rows],[x['perplexity'] for x in rows],
            marker='o',markersize=3,label=labels[method],color=colors[method])
    ax.set_xlabel('Tiempo instrumentado por método (h)'); ax.set_ylabel('Perplexity de evaluación')
    ax.grid(alpha=.2); ax.legend(); finish(fig, 'evaluation_perplexity_time')

fig, ax = plt.subplots(figsize=(3.5,2.6))
for method in ['zero','lsp']:
    rows = csv_read(campaign/method/'steps.csv')
    if rows:
        ax.plot([int(x['step']) for x in rows],[float(x['loss']) for x in rows],
                label=labels[method],color=colors[method],alpha=.65,linewidth=.5)
ax.set_xlabel('Pasos de entrenamiento'); ax.set_ylabel('Pérdida de entrenamiento (sin suavizado)')
if ax.lines: ax.legend(); ax.grid(alpha=.2); finish(fig,'training_loss')
else: plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(7.16,2.5))
for method in ['zero','lsp']:
    rows = csv_read(campaign/method/'samples.csv')
    if rows:
        hours = [float(x['elapsed_s'])/3600 for x in rows]
        axes[0].plot(hours,[int(x['rss_bytes'])/2**30 for x in rows],label=labels[method],color=colors[method],linewidth=.6)
        axes[1].plot(hours,[int(x['gpu_allocated_bytes'])/2**30 for x in rows],label=labels[method]+' allocated',color=colors[method],linewidth=.6)
        axes[1].plot(hours,[int(x['gpu_reserved_bytes'])/2**30 for x in rows],label=labels[method]+' reserved',color=colors[method],linestyle='--',linewidth=.6)
axes[0].set_ylabel('RAM RSS (GiB)'); axes[1].set_ylabel('VRAM PyTorch (GiB)')
for ax in axes:
    ax.set_xlabel('Tiempo instrumentado (h)'); ax.grid(alpha=.2)
    if ax.lines: ax.legend()
if any(ax.lines for ax in axes): finish(fig,'memory')
else: plt.close(fig)

complete = len(states)==2 and all(s['status']=='success' and s['completed_steps']==5000 for s in states.values())
report = dict(campaign=plan['campaign'], status='complete' if complete else 'partial_or_failed',
    configuration=plan, launch=read(campaign/'launch.json'), common_initial=initial_eval,
    initial_evaluation_reuse=initial_reuse,
    main_measurements=measurements, evaluations=evals, training_intervals=intervals,
    reused_microbenchmark_statistics=reuse['performance'],
    microbenchmark_version_scope='Historical commit '+plan['reuse']['performance_code_commit']+' before patch 0003; not measurements of this corrected version',
    generated_figures=generated,
    limitations=['One main trajectory per method; no estimate of quality dispersion across runs',
        '5000 steps do not establish complete convergence',
        'Medium replaces paper Large after observed RAM OOM; Zero LR1e-4 differs from paper1e-5',
        'Perplexity is exp(mean(batch loss)), with original prompt/padding semantics',
        'Plot time is measured since instrumentation start, includes calibration/evaluation; imports outside this clock appear in external_stage_s',
        'Initial evaluation shared once, plotted at step/time0; its duration is recorded separately',
        'Reused initial evaluation duration belongs to main-v1, not the wall time of this campaign',
        'Six short microbenchmarks precede patch 0003; their dispersion and speed do not characterize the corrected code',
        'RAM sampled every second; CUDA allocated/reserved is not total device memory',
        'Final weights are not an exact-resume training checkpoint'])
(output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def f(value, digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


tex = [r'\section{Resultados de LSP-Offload frente a Zero-Offload}',
       r'\label{sec:lsp-local-results}',
       r'La evaluación local utiliza GPT-2 Medium, Alpaca GPT-4, lote de cuatro secuencias, longitud máxima de 512 tokens y semilla 2026. '
       r'Se conservan el checkpoint inicial, tokenizador y particiones para ambas rutas. Los pesos usan bfloat16 y el Adam CPU original usa float32. '
       r'El descenso desde GPT-2 Large responde al OOM de RAM registrado en la inicialización de Zero. '
       r'La tasa de aprendizaje es $10^{-4}$ para ambos métodos, según el script publicado; el artículo indica $10^{-5}$ para Zero '
       r'(p.~11, Instruction Fine-tuning on Alpaca). LSP usa $d=512$, $r=8$, umbral 0.8 y comprobación cada 1000 pasos; el artículo indica umbral 0.5. '
       r'Se mantienen las correcciones autorizadas de momentos y planificación y se integra el parche 0003: '
       r'esperar H2D antes de validar delta, conservando el rechazo de NaN e infinitos. Corrige un defecto demostrado de orden de lectura; '
       r'no demuestra que este explique exclusivamente el fallo histórico de main-v1. Las fórmulas permanecen intactas.',
       r'\subsection{Rendimiento medido}',
       r'Las seis repeticiones cortas pertenecen a la versión histórica \texttt{7a57c3b}, anterior al parche 0003; '
       r'no miden el rendimiento del código corregido. Se conservan como referencia: cinco pasos de calentamiento y veinte medidos por repetición, tres por método. '
       r'La dispersión siguiente es la desviación estándar muestral entre ejecuciones; no procede de fluctuaciones de una curva.',
       r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}\caption{Microbenchmarks históricos anteriores al parche 0003: mediana y desviación estándar (DE).}\label{tab:lsp-micro}',
       r'\begin{tabular}{lrrrr}\hline Método & Med. s/it & DE s/it & Med. it/s & DE it/s \\\hline']
for method in ['zero','lsp']:
    m=reuse['performance'][method]
    tex.append(f"{labels[method]} & {f(m['median_seconds_per_iteration'])} & {f(m['std_seconds_per_iteration'])} & {f(m['median_iterations_per_second'])} & {f(m['std_iterations_per_second'])} "+r'\\')
tex += [r'\hline\end{tabular}\end{table}',
        r'Los tiempos de entrenamiento principal se miden en bloques entre evaluaciones e incluyen su vaciado final; '
        r'la calibración y la evaluación se registran por separado. Los primeros cinco pasos forman parte de los 5000 pasos, sin calentamiento adicional.']
if measurements:
    tex += [r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}\caption{Ejecución principal: tiempos reales y pasos completados.}\label{tab:lsp-main-time}',
            r'\begin{tabular}{lrrrr}\hline Método & Pasos & Entr. (s) & Cal. (s) & Eval. (s) \\\hline']
    for m in measurements:
        tex.append(f"{labels[m['method']]} & {m['completed_steps']} & {f(m['training_s'],1)} & {f(m['calibration_s'],1)} & {f(m['evaluation_s'],1)} "+r'\\')
    tex += [r'\hline\end{tabular}\end{table}',
            r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}\caption{Versión corregida: rendimiento medio del entrenamiento y tiempo total externo por método.}\label{tab:lsp-main-rate}',
            r'\begin{tabular}{lrrr}\hline Método & s/it & it/s & Total (s) \\\hline']
    for m in measurements:
        tex.append(f"{labels[m['method']]} & {f(m['seconds_per_iteration'])} & {f(m['iterations_per_second'])} & {f(m['external_stage_s'],1)} "+r'\\')
    tex += [r'\hline\end{tabular}\end{table}',r'\subsection{Calidad observada}',
            r'La perplexity conserva la definición del evaluador original: exponencial de la media de pérdidas por lote, '
            r'sin sustituirla por una media ponderada por tokens ni enmascarar de nuevo prompt o padding. '
            r'Las evaluaciones periódicas usan los 5185 ejemplos completos.']
    if initial_eval:
        tex.append(f"La evaluación inicial común produjo pérdida {f(initial_eval['mean_batch_loss'],4)} y perplexity {f(initial_eval['perplexity'],4)}, en {f(initial_eval['seconds'],1)} s, contabilizados por separado.")
        if initial_reuse:
            tex.append(r'Esta evaluación completa se reutiliza de main-v1 tras verificar los hashes de checkpoint, tokenizador y datos, el split y el protocolo. '
                       r'Su duración es histórica y no se añade al tiempo real de main-v2. Cada entrenamiento comienza desde el checkpoint preentrenado '
                       r'en un proceso independiente con su semilla inicial; no se carga estado de entrenamiento de main-v1.')
    tex += [r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}\caption{Última evaluación completa disponible.}\label{tab:lsp-main-quality}',
            r'\begin{tabular}{lrrr}\hline Método & Paso & Pérdida & Perplexity \\\hline']
    for m in measurements:
        tex.append(f"{labels[m['method']]} & {m['final_evaluated_step'] if m['final_evaluated_step'] is not None else '--'} & {f(m['final_eval_loss'],4)} & {f(m['final_eval_perplexity'],4)} "+r'\\')
    tex += [r'\hline\end{tabular}\end{table}']
if 'evaluation_perplexity_steps' in generated:
    tex += [r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{figures/evaluation_perplexity_steps.pdf}',
            r'\caption{Perplexity medida sobre el conjunto completo. Los marcadores son evaluaciones reales; las líneas solo conectan observaciones.}\label{fig:lsp-measured-quality}\end{figure}']
if 'evaluation_perplexity_time' in generated:
    tex += [r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{figures/evaluation_perplexity_time.pdf}',
            r'\caption{Perplexity frente al tiempo instrumentado real por método, incluyendo calibración y evaluación. La evaluación inicial compartida se representa en el origen; no se reconstruyen tiempos con constantes del artículo.}\label{fig:lsp-measured-time}\end{figure}']
tex += [r'\subsection{Memoria y limitaciones}',
        r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}\caption{Picos registrados durante cada ejecución (GiB).}\label{tab:lsp-main-memory}',
        r'\begin{tabular}{lrrr}\hline Método & RAM RSS & GPU asignada & GPU reservada \\\hline']
for m in measurements:
    tex.append(f"{labels[m['method']]} & {f(m['peak_rss_gib'])} & {f(m['peak_gpu_allocated_gib'])} & {f(m['peak_gpu_reserved_gib'])} "+r'\\')
tex += [r'\hline\end{tabular}\end{table}',
        r'RSS se muestrea cada segundo. La memoria CUDA asignada y reservada de PyTorch no equivale al uso total de la GPU. '
        r'Se realiza una trayectoria principal por método: no se estima dispersión de calidad entre semillas ni se afirma convergencia completa por alcanzar 5000 pasos. '
        r'El modelo reducido, el hardware local, la tasa de aprendizaje y las adaptaciones documentadas limitan la comparación directa con las cifras del artículo. '
        r'Los pesos finales, cuando están disponibles, no contienen un estado validado de reanudación exacta.']
if not complete:
    tex += [r'\textbf{Resultado parcial:} al menos una ruta no completó los 5000 pasos. Las tablas muestran únicamente mediciones disponibles; consultar results.json y los logs para el fallo concreto.']
(output/'results_ieee.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf-8')
(output/'README.txt').write_text('Measured campaign artifacts. Use results_ieee.tex inside an IEEEtran document with graphicx and UTF-8 support; copy the figures directory beside the build or adjust graphicspath. LaTeX fragment generated, not claimed as compiled. PDF/PNG figures and CSV/JSON tables contain actual measurements. Initial evaluation is shared; no exact-resume checkpoint is available.\n')
print(json.dumps(dict(status=report['status'],output=str(output),figures=generated,main_measurements=measurements),indent=2))
