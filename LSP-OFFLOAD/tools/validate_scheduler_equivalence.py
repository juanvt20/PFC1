"""Multi-step CPU-Adam equivalence, boundary drains, and reentrant checkpointing."""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

import torch
from torch.utils.checkpoint import checkpoint

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
import para_offload_modules as off
from gradient_compressor import GradientCompressor, LearnedCountSketchCompressor
from state import State

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
RTOL, ATOL = 5e-5, 2e-6
report = dict(source=args.source, scope='12_steps_6_layers_cpu_adam_same_inputs_weights_projectors',
              rtol=RTOL, atol=ATOL, diagnostic_projectors='d8/r1 Gaussian with original sampling; r1 avoids duplicate-index ambiguity',
              cases=[])


class Toy(torch.nn.Module):
    def __init__(self, layers, ckpt):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.ckpt = ckpt

    def forward(self, x):
        for layer in self.layers:
            def block(value, layer=layer):
                return torch.tanh(layer(value))
            x = checkpoint(block, x, use_reentrant=True) if self.ckpt and self.training else block(x)
        return x


def tensor_error(a, b):
    a, b = a.double(), b.double()
    diff = (a-b).abs()
    return dict(max_abs=diff.max().item(), max_scaled=(diff/(ATOL+RTOL*b.abs())).max().item(),
                passed=bool(torch.all(diff <= ATOL+RTOL*b.abs())))


def same_snapshot(a, b):
    return (a['t'] == b['t'] and a['counts'] == b['counts']
            and all(torch.equal(x, y) for key in ['W','M','V'] for x,y in zip(a[key],b[key])))


def trajectory(route, ckpt, dtype, serial):
    torch.manual_seed(337)
    state = {'state': State.RUNNING}
    layers = [off.LoRaOffloadLinear(torch.nn.Linear(16,16,bias=False).to('cuda',dtype),
              lr=1e-3,offloading_device='cpu',global_state=state,use_grad_ckpt=ckpt) for _ in range(6)]
    model = Toy(layers, ckpt)
    modules = [layer.optim for layer in layers]
    comp = (LearnedCountSketchCompressor(sketch_size=8,n_unempty=1,n_iter=2,n_ft_iter=0,
             cs_init='gaussian',thresh_hold=-1) if route=='lsp' else GradientCompressor())
    comp.fit(modules)
    for m in modules:
        m.gradient_compressor = comp
    settings = (dict(zero=False,fcfs_point=0) if serial else
                dict(zero=True) if route=='zero' else
                dict(zero=False,fcfs_point=3,fcfs_process_delay=1,fcfs_h2d_delay=1,
                     fcfs_upd_delay=1,lcfs_d2h_delay=99,lcfs_process_delay=1,lcfs_h2d_delay=0))
    scheduler = off.Scheduler(modules,**settings)
    generated, applied = {id(m.ctx):0 for m in modules}, {id(m.ctx):[] for m in modules}
    update_original = off._update

    def counted_update(ctx,*pos,**kwargs):
        fn = update_original(ctx,*pos,**kwargs)
        generated[id(ctx)] += 1
        token = generated[id(ctx)]
        def run():
            if token in applied[id(ctx)]:
                raise AssertionError('Duplicate weight update')
            fn()
            applied[id(ctx)].append(token)
        return run

    def snapshot():
        torch.cuda.synchronize()
        return dict(W=[l.weight.detach().cpu().clone() for l in layers],
                    M=[m.optim.m.cpu().clone() for m in modules],
                    V=[m.optim.v.cpu().clone() for m in modules],
                    t=[m.optim.t for m in modules],counts=[list(applied[id(m.ctx)]) for m in modules])

    snapshots, checks, losses = {}, [], []
    g = torch.Generator().manual_seed(920)
    inputs = [torch.randn(4,16,generator=g).to('cuda',dtype) for _ in range(13)]
    targets = [torch.randn(4,16,generator=g).to('cuda',dtype) for _ in range(13)]
    off._update = counted_update
    try:
        with off.ScheduleEnv(scheduler):
            for step in range(1,13):
                model.train()
                x=inputs[step-1].detach().requires_grad_(ckpt)
                loss=(model(x)-targets[step-1]).square().mean()
                if not torch.isfinite(loss):
                    raise AssertionError('Non-finite loss')
                loss.backward()
                losses.append(loss.item())
                if step == 1 and not serial:
                    pending_before = copy.deepcopy(scheduler.pending_work())
                    rejected = False
                    try:
                        scheduler.clear()
                    except RuntimeError:
                        rejected = True
                    checks.append(dict(name='clear_rejects_without_discard',passed=rejected
                                       and bool(pending_before) and pending_before==scheduler.pending_work()))
                if step not in (3,6,9,12):
                    continue  # Never drain ordinary iterations.
                boundary={3:'evaluation',6:'recalibration',9:'save',12:'exit'}[step]
                if boundary!='exit':
                    scheduler.drain()
                else:
                    # Exercise the same context manager path as the trainer.
                    with off.ScheduleEnv(scheduler):
                        pass
                snap=snapshot()
                snapshots[boundary]=snap
                checks.append(dict(name=boundary+'_exactly_once',passed=snap['t']==[step]*6
                    and all(x==list(range(1,step+1)) for x in snap['counts']) and not scheduler.pending_work()))
                scheduler.drain()
                checks.append(dict(name=boundary+'_idempotent',passed=same_snapshot(snap,snapshot())))
                if boundary=='evaluation':
                    model.eval()
                    with torch.no_grad():
                        value=(model(inputs[-1])-targets[-1]).square().mean()
                    checks.append(dict(name='evaluation_no_update',passed=torch.isfinite(value).item()
                                       and same_snapshot(snap,snapshot())))
                elif boundary=='recalibration' and route=='lsp':
                    for m in modules:
                        m.to('cpu')
                    state['state']=State.RUNNING_PROFILE
                    model.train()
                    model(inputs[-1].detach().requires_grad_(ckpt)).square().mean().backward()
                    model.cpu()
                    torch.manual_seed(609)
                    comp.fit(modules)
                    model.cuda()
                    state['state']=State.RUNNING
                    snapshots['after_recalibration']=snapshot()
                    checks.append(dict(name='recalibration_does_not_increment_steps',
                                       passed=snap['t']==snapshot()['t']))
                elif boundary=='save':
                    torch.save(snap,out/f'{route}-{ckpt}-{dtype}-{serial}-saved.pt')
                    checks.append(dict(name='save_no_update',passed=same_snapshot(snap,snapshot())))
        checks.append(dict(name='context_exit_idempotent',passed=same_snapshot(snapshots['exit'],snapshot())))
    finally:
        off._update=update_original
    torch.save(snapshots,out/f'{route}-{ckpt}-{dtype}-{serial}-states.pt')
    return snapshots, checks, losses, settings


for route in ['zero','lsp']:
    for ckpt in [False,True]:
        for dtype in [torch.float32,torch.bfloat16]:
            row=dict(route=route,checkpointing=ckpt,dtype=str(dtype))
            try:
                reference,ref_checks,ref_losses,_=trajectory(route,ckpt,dtype,True)
                candidate,checks,losses,settings=trajectory(route,ckpt,dtype,False)
                comparisons=[]
                for boundary,snap in candidate.items():
                    other=reference[boundary]
                    comparisons.append(dict(boundary=boundary,
                        counters_equal=snap['t']==other['t'] and snap['counts']==other['counts'],
                        tensors={key:[tensor_error(x,y) for x,y in zip(snap[key],other[key])]
                                 for key in ['W','M','V']}))
                row.update(checks=checks,reference_checks=ref_checks,comparisons=comparisons,
                           candidate_losses=losses,reference_losses=ref_losses,scheduler=settings)
                row['passed']=all(x['passed'] for x in checks+ref_checks) and all(
                    c['counters_equal'] and all(x['passed'] for values in c['tensors'].values()
                                              for x in values) for c in comparisons)
            except Exception:
                row.update(passed=False,error=traceback.format_exc())
            report['cases'].append(row)
            (out/'diagnostics.json').write_text(json.dumps(report,indent=2))
report['passed']=all(x['passed'] for x in report['cases'])
(out/'diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=report['passed'],cases=[{k:v for k,v in c.items() if k in
    ['route','checkpointing','dtype','passed','error']} for c in report['cases']]),indent=2))
raise SystemExit(not report['passed'])
