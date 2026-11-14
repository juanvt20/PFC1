"""Small offline diagnostics of original code; no training or numerical patches."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

root = Path('/work')
sys.path.insert(0, str(root / 'original/upstream'))
parser = argparse.ArgumentParser()
parser.add_argument('--run-id', default='environment-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
args = parser.parse_args()
if Path(args.run_id).name != args.run_id:
    parser.error('--run-id must be a directory name, not a path')
out = root / 'results' / args.run_id
out.mkdir(parents=True, exist_ok=False)
report = dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              scope='offline_small_diagnostics_no_training', python=platform.python_version(),
              torch=torch.__version__, cuda=torch.version.cuda, imports={}, kernels=[],
              algorithm_accepted=False,
              algorithm_blockers=['N01 epsilon discrepancy', 'N15 moments reset',
                                  'N04 objective/first-fit mismatch', 'scheduler correctness pending'])
def run(name, fn):
    try:
        report[name] = fn()
    except Exception:
        report[name] = dict(error=traceback.format_exc())
    finally:
        (out / 'diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

def environment():
    p = torch.cuda.get_device_properties(0)
    info = dict(gpu=p.name, capability=torch.cuda.get_device_capability(0),
                vram_free_total_bytes=torch.cuda.mem_get_info(), cpu_count=os.cpu_count(),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                cpu_model=next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                               if s.startswith('model name')),
                avx2='avx2' in Path('/proc/cpuinfo').read_text(),
                meminfo=[s for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith(('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:'))])
    for n in ['memory.max', 'memory.current', 'cpu.max']:
        info[n] = Path('/sys/fs/cgroup', n).read_text().strip()
    for cmd in [['nvcc', '--version'], ['c++', '--version']]:
        info[cmd[0]] = subprocess.check_output(cmd).decode()
    return info

def imports():
    for name in ['transformers', 'datasets', 'evaluate', 'diskcache', 'sklearn',
                 'gradient_compressor', 'para_offload_modules', 'llm_finetuning']:
        try:
            m = importlib.import_module(name)
            report['imports'][name] = dict(ok=True, version=getattr(m, '__version__', None))
        except Exception:
            report['imports'][name] = dict(ok=False, error=traceback.format_exc())
    return all(v['ok'] for v in report['imports'].values())

def kernels():
    from kernels import fused_adam
    rows = []
    for device in ['cpu', 'cuda']:
        m = torch.zeros(32, device=device)
        v = torch.zeros_like(m)
        mr, vr = m.clone(), v.clone()
        for step in range(1, 4):
            g = torch.linspace(-0.02, 0.02, 32, device=device)
            mr = 0.9 * mr + 0.1 * g
            vr = 0.999 * vr + 0.001 * g.square()
            mh, vh = mr / (1 - 0.9 ** step), vr / (1 - 0.999 ** step)
            denom = (vh + 1e-4).sqrt() if device == 'cpu' else vh.sqrt() + 1e-4
            expected = -1e-3 * mh / denom
            fused_adam.fused_adam(g, m, v, 0.9, 0.999, 1e-3, 1e-4, step)
            if device == 'cuda':
                torch.cuda.synchronize()
            errors = dict(delta=(g-expected).abs().max().item(),
                          m=(m-mr).abs().max().item(), v=(v-vr).abs().max().item())
            rows.append(dict(device=device, step=step, max_abs_errors=errors,
                             passed=max(errors.values()) < 2e-6))
    return rows

def pinned_transfer():
    src = torch.arange(262144, dtype=torch.float32).pin_memory()
    dst = torch.empty_like(src, device='cuda')
    back = torch.empty_like(src, pin_memory=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dst.copy_(src, non_blocking=True)
        back.copy_(dst, non_blocking=True)
    stream.synchronize()
    return dict(bytes_each_direction=src.numel()*src.element_size(),
                pinned=src.is_pinned() and back.is_pinned(), equal=torch.equal(src, back),
                claim='correctness_only_not_bandwidth_or_overlap')

def compressor():
    from gradient_compressor import CountSketch, LearnedCountSketchCompressor
    from para_offload_modules import ParallelOffloadAdamMMOptim
    torch.manual_seed(1234)
    sketch = CountSketch(8, 128, 4, 'cuda', torch.float32, 'binary')
    binary_values = sketch.values.unique().tolist()
    distinct = [len(set(col)) for col in sketch.indices.T.cpu().tolist()]
    weight = torch.randn(16, 16, device='cuda')
    module = ParallelOffloadAdamMMOptim(weight, 1e-3, offloading_device='cuda:0', global_state={})
    comp = LearnedCountSketchCompressor(sketch_size=8, n_unempty=2, n_iter=2,
                                       n_ft_iter=0, cs_init='gaussian', thresh_hold=-1)
    module._grad = torch.randn_like(weight).cpu()
    comp.fit([module])
    before = comp.sketches_map[module][0].values.detach().clone()
    module.optim.m.fill_(1)
    module.optim.v.fill_(2)
    init_original = module.init
    pre_init = {}
    def capture_then_init(*args):
        pre_init['m_abs_sum'] = module.optim.m.abs().sum().item()
        pre_init['v_abs_sum'] = module.optim.v.abs().sum().item()
        return init_original(*args)
    module.init = capture_then_init
    module._grad = torch.randn_like(weight).cpu()
    comp.fit([module])
    after = comp.sketches_map[module][0].values.detach()
    return dict(binary_values=binary_values, expected_random_signs=False,
                columns_with_collisions=sum(n<4 for n in distinct),
                projector_changed_on_second_fit=not torch.equal(before, after),
                transformed_moments_before_init=pre_init,
                second_fit_m_abs_sum=module.optim.m.abs().sum().item(),
                second_fit_v_abs_sum=module.optim.v.abs().sum().item(),
                interpretation='moments must be inspected; zeros after second fit indicate reset, not preserved transport',
                diagnostic_config=dict(d=8, r=2, shape=[16,16], n_iter=2, n_ft_iter=0,
                                       threshold=-1, dtype='float32', seed=1234))

run('environment', environment)
run('imports_ok', imports)
run('kernels', kernels)
run('pinned_transfer', pinned_transfer)
run('compressor', compressor)
subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(out/'environment.freeze.txt').open('w'))
print(json.dumps(report, indent=2))
failed = not report.get('imports_ok') or any(isinstance(report.get(k), dict) and 'error' in report[k]
    for k in ['environment', 'kernels', 'pinned_transfer', 'compressor'])
failed = failed or not all(r['passed'] for r in report.get('kernels', []) if isinstance(r, dict))
raise SystemExit(bool(failed))
