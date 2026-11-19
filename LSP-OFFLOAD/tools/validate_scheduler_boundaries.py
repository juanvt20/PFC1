"""Reproduce lost updates at actual evaluation/exit boundaries, without a fix."""
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
from gradient_compressor import GradientCompressor, LearnedCountSketchCompressor
from para_offload_modules import LoRaOffloadLinear, Scheduler, ScheduleEnv
from state import State

torch.set_num_threads(2)
report = dict(source=args.source, scope='two_layer_actual_cpu_adam_no_model_training', cases=[])


def case(route, boundary):
    torch.manual_seed(2026)
    state = {'state': State.RUNNING}
    layers = [LoRaOffloadLinear(torch.nn.Linear(16, 16, bias=False).cuda(), lr=1e-3,
                               offloading_device='cpu', global_state=state) for _ in range(2)]
    model = torch.nn.Sequential(layers[0], torch.nn.Tanh(), layers[1])
    modules = [layer.optim for layer in layers]
    comp = (LearnedCountSketchCompressor(sketch_size=8, n_unempty=2,
                n_iter=2, n_ft_iter=0, cs_init='gaussian') if route == 'lsp_layerwise'
            else GradientCompressor())
    comp.fit(modules)
    for module in modules:
        module.gradient_compressor = comp
    settings = dict(zero=route == 'zero', fcfs_point=1 if route == 'lsp_layerwise' else 0,
                    fcfs_process_delay=0, fcfs_h2d_delay=0, fcfs_upd_delay=0,
                    lcfs_d2h_delay=1 if route == 'lsp_layerwise' else 0,
                    lcfs_process_delay=0, lcfs_h2d_delay=0)
    scheduler = Scheduler(modules, **settings)
    before = [layer.weight.detach().clone() for layer in layers]
    x = torch.randn(2, 16, device='cuda')
    target = torch.randn(2, 16, device='cuda')

    def snapshot():
        torch.cuda.synchronize()
        return dict(adam_steps=[m.optim.t for m in modules],
                    pending_hooks=[sum(len(getattr(m, name)) for name in
                        ['fwd_hooks', 'fwd_end_hooks', 'bwd_hooks', 'bwd_end_hooks']) for m in modules],
                    context_stack=[len(m.ctx.stack) for m in modules],
                    weight_max_change=[(layer.weight - old).abs().max().item()
                                       for layer, old in zip(layers, before)])

    with ScheduleEnv(scheduler):
        prediction = model(x)
        loss = (prediction - target).square().mean()
        loss.backward()
        after_backward = snapshot()
        if boundary == 'evaluation':
            # Exactly the ordering in Trainer.train_epoch before evaluation.
            scheduler.clear()
            model.eval()
            with torch.no_grad():
                evaluation_loss = (model(x) - target).square().mean().item()
        else:
            evaluation_loss = None
    after_boundary = snapshot()  # ScheduleEnv.__exit__ has called clear().
    torch.save(dict(before=[w.cpu() for w in before],
                    after=[layer.weight.detach().cpu() for layer in layers],
                    m=[m.optim.m.cpu() for m in modules], v=[m.optim.v.cpu() for m in modules]),
               out / f'{route}-{boundary}.pt')
    return dict(route=route, boundary=boundary, scheduler=settings,
                cpu_optimizer='unchanged para_offload_modules.Adam, eps=1e-4',
                loss=loss.item(), loss_finite=bool(torch.isfinite(loss)),
                evaluation_loss=evaluation_loss, after_backward=after_backward,
                after_boundary=after_boundary, expected_adam_steps=[1, 1],
                passed=(after_boundary['adam_steps'] == [1, 1]
                        and after_boundary['pending_hooks'] == [0, 0]
                        and after_boundary['context_stack'] == [0, 0]
                        and all(x > 0 for x in after_boundary['weight_max_change'])))


for route in ['serial_control', 'zero', 'lsp_layerwise']:
    for boundary in ['evaluation', 'exit']:
        try:
            report['cases'].append(case(route, boundary))
        except Exception:
            report['cases'].append(dict(route=route, boundary=boundary, passed=False,
                                        error=traceback.format_exc()))
        (out / 'diagnostics.json').write_text(json.dumps(report, indent=2))
report['passed'] = all(x['passed'] for x in report['cases'])
(out / 'diagnostics.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(not report['passed'])
