"""Check full transported M/V against Algorithm 1, p.5, lines 8-9.

Runs original or adapted sources without modifying them. It intentionally
forces two small recalibrations; this is not a model-training result.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', choices=['original', 'adaptado'], required=True)
parser.add_argument('--run-id', required=True)
args = parser.parse_args()
root = Path('/work')
if Path(args.run_id).name != args.run_id:
    parser.error('run-id must be a directory name')
out = root / 'results' / args.run_id
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root / args.source / 'upstream'))
from gradient_compressor import CountSketch, LearnedCountSketchCompressor
from para_offload_modules import ParallelOffloadAdamMMOptim

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
report = dict(source=args.source, paper='p.5 Algorithm 1 lines 8-9',
              scope='full_tensor_moment_transport_not_training',
              reference_inputs='actual dense sketches returned during transport; no resampling or duplicate-index correction',
              cases=[])


def comparison(actual, expected, rtol=2e-5, atol=2e-6):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    diff = (actual - expected).abs()
    return dict(shape=list(actual.shape), max_abs_error=diff.max().item(),
                max_scaled_error=(diff / (atol + rtol * expected.abs())).max().item(),
                all_elements_pass=bool(torch.all(diff <= atol + rtol * expected.abs())),
                rtol=rtol, atol=atol)


def case(device, dtype):
    torch.manual_seed(1234)
    weight = torch.randn(16, 24, device='cuda', dtype=dtype)
    module = ParallelOffloadAdamMMOptim(weight, 1e-3, offloading_device=device,
                                       optim_dtype=torch.float32, global_state={})
    comp = LearnedCountSketchCompressor(sketch_size=8, n_unempty=2,
        n_iter=2, n_ft_iter=0, cs_init='gaussian', thresh_hold=-1)
    module._grad = torch.randn(16, 24)
    comp.fit([module])
    initial = dict(m_zero=bool(torch.count_nonzero(module.optim.m) == 0),
                   v_zero=bool(torch.count_nonzero(module.optim.v) == 0),
                   step_zero=module.optim.t == 0, device=str(module.optim.m.device),
                   buffer_pinned=module.pinned_buffer_grad.is_pinned() if device == 'cpu' else None)
    initial['passed'] = (initial['m_zero'] and initial['v_zero'] and initial['step_zero']
                         and initial['device'] == device
                         and (device != 'cpu' or initial['buffer_pinned']))
    module.optim.m.copy_(torch.linspace(-.2, .3, 64, device=device).reshape(8, 8))
    module.optim.v.copy_(torch.linspace(.1, .5, 64, device=device).reshape(8, 8))
    module.optim.t = 7
    rows = []
    for recalibration in range(1, 3):
        old_sketch_l, old_sketch_r = comp.sketches_map[module]
        old_m = module.optim.m.detach().cpu().clone()
        old_v = module.optim.v.detach().cpu().clone()
        old_t = module.optim.t
        module.to('cpu')  # Same temporary state move as Trainer.train_epoch.
        module._grad = torch.randn(16, 24)
        init_original = module.init
        captured = {}
        dense_used = {}
        get_sketch_original = CountSketch.get_sketch

        def observe_sketch(sketch, train=False):
            value = get_sketch_original(sketch, train=train)
            if not train:
                dense_used[id(sketch)] = value.detach().clone()
            return value

        def observe_init(*pos, **kwargs):
            captured['pre_m'] = module.optim.m.detach().cpu().clone()
            captured['pre_v'] = module.optim.v.detach().cpu().clone()
            return init_original(*pos, **kwargs)

        module.init = observe_init
        CountSketch.get_sketch = observe_sketch
        try:
            comp.fit([module])
        finally:
            module.init = init_original
            CountSketch.get_sketch = get_sketch_original
        new_l, new_r = [dense_used[id(s)] for s in comp.sketches_map[module]]
        old_l, old_r = [dense_used[id(s)] for s in [old_sketch_l, old_sketch_r]]
        # Upstream stores P^T/Q^T. Preserve its weight-dtype rounding in
        # the overlaps, then independently evaluate transport in float64.
        a = (new_l @ old_l.T).float().cpu().double()
        b = (new_r @ old_r.T).float().cpu().double()
        expected_m = a @ old_m.double() @ b.T
        expected_v = a.square() @ old_v.double() @ b.T.square()
        actual_m = module.optim.m.detach().cpu().clone()
        actual_v = module.optim.v.detach().cpu().clone()
        row = dict(recalibration=recalibration, step_before=old_t,
                   step_after=module.optim.t, state_device=str(module.optim.m.device),
                   projector_changed=not torch.equal(new_l, old_l),
                   pre_m=comparison(captured['pre_m'], expected_m),
                   pre_v=comparison(captured['pre_v'], expected_v),
                   post_m=comparison(actual_m, expected_m),
                   post_v=comparison(actual_v, expected_v),
                   buffer_preserves_m=torch.equal(captured['pre_m'], actual_m),
                   buffer_preserves_v=torch.equal(captured['pre_v'], actual_v))
        torch.save(dict(old_m=old_m, old_v=old_v, old_l=old_l.cpu(), old_r=old_r.cpu(),
                        new_l=new_l.cpu(), new_r=new_r.cpu(), overlap_left=a, overlap_right=b,
                        expected_m=expected_m, expected_v=expected_v,
                        **captured, actual_m=actual_m, actual_v=actual_v),
                   out / f'{device.replace(":", "-")}-{str(dtype).split(".")[-1]}-{recalibration}.pt')
        # One subsequent step, using each route's unchanged epsilon formula.
        g = torch.linspace(-.02, .03, 64, device=device).reshape(8, 8).contiguous()
        mr = .9 * actual_m + .1 * g.cpu()
        vr = .999 * actual_v + .001 * g.cpu().square()
        t = old_t + 1
        vh = vr / (1 - .999**t)
        denom = (vh + 1e-4).sqrt() if device == 'cpu' else vh.sqrt() + 1e-4
        expected_delta = -1e-3 * (mr / (1 - .9**t)) / denom
        module.optim.step(g)
        torch.cuda.synchronize()
        row['next_step'] = dict(m=comparison(module.optim.m, mr), v=comparison(module.optim.v, vr),
                                delta=comparison(g, expected_delta), counter_ok=module.optim.t == t)
        row['passed'] = all(row[k]['all_elements_pass'] for k in ['pre_m', 'pre_v', 'post_m', 'post_v'])
        row['passed'] &= all(row['next_step'][k]['all_elements_pass'] for k in ['m', 'v', 'delta'])
        row['passed'] &= (row['buffer_preserves_m'] and row['buffer_preserves_v']
                          and row['state_device'] == device and row['step_after'] == old_t
                          and row['projector_changed'] and row['next_step']['counter_ok'])
        rows.append(row)
    return dict(device=device, weight_dtype=str(dtype), initial=initial, recalibrations=rows,
                passed=initial['passed'] and all(x['passed'] for x in rows))


for device in ['cpu', 'cuda:0']:
    for dtype in [torch.float32, torch.bfloat16]:
        try:
            report['cases'].append(case(device, dtype))
        except Exception:
            report['cases'].append(dict(device=device, dtype=str(dtype), passed=False,
                                        error=traceback.format_exc()))
        (out / 'diagnostics.json').write_text(json.dumps(report, indent=2))
report['passed'] = all(x['passed'] for x in report['cases'])
(out / 'diagnostics.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(not report['passed'])
