"""PFC1: lanza el pretrain_gpt.py ORIGINAL con datos WikiText-103 reales.

No reimplementa GPT ni STRONGHOLD. Construye argumentos del artefacto
SC22 y registra el intento en un JSON del volumen /results.
"""

import argparse
import json
import math
import os
import runpy
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/wikitext103')


def save_json(path, value):
    tmp = path.with_suffix('.json.partial')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    tmp.replace(path)


def parse_pfc_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['baseline', 'stronghold'], required=True)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--seq-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--iters', type=int, default=2)
    p.add_argument('--gl-window-size', type=int, default=2)
    p.add_argument('--output-dir', default='/results/STRONGHOLD')
    p.add_argument('--data-root', default=str(DATA))
    args = p.parse_args()
    if args.layers < 2 or args.hidden_size < 64 or args.heads < 1:
        p.error('Capas, dimension oculta o cabezas no validas')
    if args.hidden_size % args.heads:
        p.error('--hidden-size debe ser divisible entre --heads')
    if args.seq_length < 16 or args.iters < 1 or args.batch_size < 1:
        p.error('Secuencia, pasos y batch deben ser positivos')
    if args.mode == 'stronghold' and not 2 <= args.gl_window_size <= args.layers:
        p.error('STRONGHOLD requiere una ventana GPU entre 2 y el numero de capas')
    return args


def megatron_argv(a):
    data_root = Path(a.data_root)
    prefix = data_root / 'megatron' / 'wikitext103_gpt2_text_document'
    vocab = data_root / 'tokenizer' / 'gpt2-vocab.json'
    merges = data_root / 'tokenizer' / 'gpt2-merges.txt'
    for path in (Path(str(prefix) + '.bin'), Path(str(prefix) + '.idx'), vocab, merges):
        if not path.is_file():
            raise FileNotFoundError(f'Falta archivo de datos: {path}')

    argv = [
        'pretrain_gpt.py',
        '--num-layers', str(a.layers),
        '--hidden-size', str(a.hidden_size),
        '--num-attention-heads', str(a.heads),
        '--seq-length', str(a.seq_length),
        '--max-position-embeddings', str(a.seq_length),
        '--micro-batch-size', str(a.batch_size),
        '--global-batch-size', str(a.batch_size),
        '--train-iters', str(a.iters),
        '--log-interval', '1',
        '--lr', '0.00015',
        '--min-lr', '0.00001',
        '--lr-decay-style', 'cosine',
        '--lr-decay-iters', '50',
        '--weight-decay', '0.01',
        '--clip-grad', '1.0',
        '--lr-warmup-fraction', '0.01',
        '--data-path', str(prefix),
        '--vocab-file', str(vocab),
        '--merge-file', str(merges),
        '--data-impl', 'mmap',
        '--split', '949,50,1',
        '--distributed-backend', 'nccl',
        '--seed', '1234',
        '--num-workers', '0',
        '--eval-interval', '1000',
        '--eval-iters', '1',
        '--checkpoint-activations',
        '--activations-checkpoint-method', 'uniform',
        '--activations-checkpoint-num-layers', '1',
    ]
    if a.mode == 'stronghold':
        argv += [
            '--enable-gl',
            '--use-cpu-initialization',
            '--gl-world-size', '1',
            '--gl-window-size', str(a.gl_window_size),
            '--gl-ray-max-concurrency', '4',
        ]
    return argv


def main():
    a = parse_pfc_args()
    os.chdir(ROOT)
    os.environ.setdefault('PYTORCH_JIT', '0')

    # Unica compatibilidad de importacion con PyTorch 2.x y DeepSpeed 0.5.8.
    # No modifica gradientes, pesos ni el algoritmo de entrenamiento.
    if 'torch._six' not in sys.modules:
        old_six = types.ModuleType('torch._six')
        old_six.inf = math.inf
        sys.modules['torch._six'] = old_six

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU CUDA no visible: inicia Docker con --gpus all')
    for env_key in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE'):
        if env_key not in os.environ:
            raise RuntimeError(f'Falta variable distribuida: {env_key}')
    if os.environ['WORLD_SIZE'] != '1' or os.environ['RANK'] != '0':
        raise RuntimeError('Este lanzador inicial admite solo una GPU/rank')

    upstream_args = megatron_argv(a)
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_id = f'{a.mode}_GPT_L{a.layers}_H{a.hidden_size}_S{a.seq_length}_{stamp}'
    manifest = outdir / f'{run_id}.json'
    event = {
        'run_id': run_id,
        'estado': 'iniciado',
        'tipo': 'smoke_test_entrenamiento_original' if a.iters <= 5 else 'entrenamiento',
        'metodo': a.mode,
        'fuente': 'STRONGHOLD SC22 adaptacion de compatibilidad; pretrain_gpt.py original',
        'dataset': 'Salesforce/wikitext wikitext-103-raw-v1 (train original)',
        'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
        'configuracion_megatron': upstream_args[1:],
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
        'fecha_inicio_utc': datetime.now(timezone.utc).isoformat(),
        'advertencia': 'En este piloto no se guardan checkpoints ni se reportan aceleraciones.',
    }
    save_json(manifest, event)
    print('=== PRIMER ENTRENAMIENTO GPT PFC1 ===', flush=True)
    print('MODO:', a.mode, flush=True)
    print('GPU:', event['gpu'], flush=True)
    print('CONFIGURACION:', ' '.join(upstream_args), flush=True)
    print('MANIFIESTO:', manifest, flush=True)
    print('Sin modificaciones del modelo GPT.', flush=True)

    old_argv = sys.argv
    started = time.monotonic()
    try:
        sys.argv = upstream_args
        runpy.run_path(str(ROOT / 'pretrain_gpt.py'), run_name='__main__')
    except BaseException as exc:
        event['estado'] = 'error'
        event['error'] = f'{type(exc).__name__}: {exc}'
        raise
    else:
        event['estado'] = 'completado'
        print('ENTRENAMIENTO GPT PFC1 COMPLETADO', flush=True)
    finally:
        sys.argv = old_argv
        event['duracion_segundos_total'] = time.monotonic() - started
        event['fecha_fin_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(manifest, event)
        print('MANIFIESTO FINAL:', manifest, flush=True)


if __name__ == '__main__':
    main()
