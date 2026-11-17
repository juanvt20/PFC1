"""Directed H2D readiness test using real _update and CPU Adam; no training experiment."""
import argparse
import difflib
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
assert Path(a.run_id).name==a.run_id
root=Path('/work');out=root/'results'/a.run_id;out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'adaptado/upstream'))
import para_offload_modules as off
from state import State
from zero_nan_trace import statistics
torch.set_num_threads(14)
source=Path(off.__file__).read_bytes()
old=b'''                assert torch.isnan(delta).sum().item() == 0
                torch.cuda.current_stream().wait_event(event)
                delta.record_stream(torch.cuda.current_stream())
'''
new=b'''                torch.cuda.current_stream().wait_event(event)
                delta.record_stream(torch.cuda.current_stream())
                assert torch.isfinite(delta).all().item()
'''
assert source.count(old)==1
patched=source.replace(old,new)
patch=''.join(difflib.unified_diff(source.decode().splitlines(True),patched.decode().splitlines(True),
    fromfile='a/para_offload_modules.py',tofile='b/para_offload_modules.py'))
(out/'candidate.patch').write_text(patch)
function=inspect.getsource(off._update)
assert function.encode().count(old)==1
namespace={}
exec(compile(function.encode().replace(old,new),'<candidate-update>','exec'),off.__dict__,namespace)
candidate=namespace['_update']
report=dict(scope='controlled delayed H2D; stale NaN is a buffer sentinel, not a model gradient',
            source_sha256=hashlib.sha256(source).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            cases=[],state_tolerance={'atol':0,'rtol':0})


def payload(source_cpu,delay):
    delta=torch.full_like(source_cpu,float('nan'),device='cuda')
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if delay: torch.cuda._sleep(100_000_000)
        delta.copy_(source_cpu,non_blocking=True)
        event=torch.cuda.Event();event.record(stream)
    ctx=off.Context();ctx.idx=0;ctx.push((event,delta))
    return ctx,event,delta,stream


for dtype in [torch.float32,torch.bfloat16]:
    for name,update in [('baseline',off._update),('candidate',candidate)]:
        optim=off.Adam(lr=1e-4,device='cpu',dtype=torch.float32);optim.init((32,32))
        serial=off.Adam(lr=1e-4,device='cpu',dtype=torch.float32);serial.init((32,32))
        weight=torch.zeros((32,32),device='cuda',dtype=dtype)
        expected=torch.zeros((32,32),dtype=dtype)
        generator=torch.Generator().manual_seed(2026)
        records=[]
        for step in range(1,9):
            grad=torch.randn((32,32),generator=generator).pin_memory()
            control=grad.clone();serial.step(control);optim.step(grad)
            assert torch.equal(grad,control)
            ctx,event,delta,stream=payload(grad,True)
            ready_before=event.query();rejected=False
            try: update(ctx,weight,lambda t:t,State.RUNNING)()
            except AssertionError: rejected=True
            event.synchronize();torch.cuda.synchronize()
            if not rejected: expected.add_(control.to(dtype))
            records.append(dict(step=step,event_ready_before=ready_before,rejected=rejected,
                settled_delta_finite=bool(torch.isfinite(delta).all().item()),
                settled_delta_matches_cpu=torch.equal(delta.cpu(),grad),
                weight_matches_serial=torch.equal(weight.cpu(),expected),
                M_matches_serial=torch.equal(optim.m,serial.m),V_matches_serial=torch.equal(optim.v,serial.v),
                adam_t=optim.t,serial_t=serial.t))
            if rejected: break
        expected_rejection=name=='baseline'
        passed=(any(x['rejected'] for x in records) if expected_rejection else len(records)==8 and not any(x['rejected'] for x in records))
        passed=passed and all(not x['event_ready_before'] and x['settled_delta_finite'] and x['settled_delta_matches_cpu']
            and x['weight_matches_serial'] and x['M_matches_serial'] and x['V_matches_serial'] and x['adam_t']==x['serial_t'] for x in records)
        report['cases'].append(dict(test='finite_cpu_delta_delayed_h2d',variant=name,weight_dtype=str(dtype),passed=passed,records=records))

for bad,label in [(float('nan'),'nan'),(float('inf'),'positive_inf'),(float('-inf'),'negative_inf')]:
    source_cpu=torch.full((32,32),bad,dtype=torch.float32,pin_memory=True)
    for name,update in [('baseline',off._update),('candidate',candidate)]:
        weight=torch.zeros((32,32),device='cuda')
        ctx,event,delta,stream=payload(source_cpu,False);event.synchronize()
        rejected=False
        try: update(ctx,weight,lambda t:t,State.RUNNING)()
        except AssertionError: rejected=True
        torch.cuda.synchronize()
        row=dict(test='completed_h2d_nonfinite_payload',variant=name,payload=label,rejected=rejected,
            weight_unchanged=bool((weight==0).all().item()),gpu_delta=statistics(delta))
        row['passed']=(rejected and row['weight_unchanged']) if name=='candidate' or label=='nan' else not rejected
        report['cases'].append(row)
report['passed']=all(x['passed'] for x in report['cases'])
(out/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
raise SystemExit(not report['passed'])
