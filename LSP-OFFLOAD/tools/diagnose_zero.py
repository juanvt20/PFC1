"""Replay only the required prefix of the exact saved main-v1 runner, with observers."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys
import types

p=argparse.ArgumentParser()
p.add_argument('--run-id',required=True)
p.add_argument('--variant',choices=['baseline','candidate'],default='baseline')
p.add_argument('--trace-from',type=int,default=150)
p.add_argument('--max-steps',type=int,default=166)
a=p.parse_args()
assert Path(a.run_id).name==a.run_id and 1<=a.max_steps<=170
root=Path('/work');main=root/'results/scheduler-v3-main-v1';out=root/'results'/a.run_id
out.mkdir(exist_ok=False)
(out/'initial').mkdir()
for name in ['plan.json','launch.json']: (out/name).write_bytes((main/name).read_bytes())
(out/'initial/summary.json').write_bytes((main/'initial/summary.json').read_bytes())
plan=json.loads((main/'plan.json').read_text())
source=root/'adaptado/upstream/para_offload_modules.py'
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()==plan['immutable_reference_hashes']['adaptado/upstream/para_offload_modules.py']
sys.path.insert(0,str(root/'adaptado/upstream'))
if a.variant=='candidate':
    old=b'''                assert torch.isnan(delta).sum().item() == 0
                torch.cuda.current_stream().wait_event(event)
                delta.record_stream(torch.cuda.current_stream())
'''
    new=b'''                torch.cuda.current_stream().wait_event(event)
                delta.record_stream(torch.cuda.current_stream())
                assert torch.isfinite(delta).all().item()
'''
    assert raw.count(old)==1
    patched=raw.replace(old,new)
    (out/'candidate-para_offload_modules.py').write_bytes(patched)
    patch=''.join(difflib.unified_diff(raw.decode().splitlines(True),patched.decode().splitlines(True),
                    fromfile='a/para_offload_modules.py',tofile='b/para_offload_modules.py'))
    (out/'candidate.patch').write_text(patch)
    module=types.ModuleType('para_offload_modules');module.__file__=str(out/'candidate-para_offload_modules.py')
    sys.modules[module.__name__]=module
    exec(compile(patched,module.__file__,'exec'),module.__dict__)
from zero_nan_trace import Trace
trace=Trace(out,a.trace_from,initial_scan=a.trace_from<=a.max_steps)
sources=out/'sources';sources.mkdir()
for name in ['diagnose_zero.py','zero_nan_trace.py','run_zero_diagnostic.ps1']:
    (sources/name).write_bytes((root/'tools'/name).read_bytes())
code=(main/'code/run_main.py').read_text()
launch=json.loads((main/'launch.json').read_text(encoding='utf-8-sig'))
assert hashlib.sha256((main/'code/run_main.py').read_bytes()).hexdigest()==launch['script_hashes']['run_main.py']

def replace(old,new):
    global code
    assert code.count(old)==1,old
    code=code.replace(old,new)

replace('def inputs(batch):', '''_get_item = Alpaca.__getitem__
def _observed_item(self,index):
    DIAG.sample(self.indices[index])
    return _get_item(self,index)
Alpaca.__getitem__ = _observed_item

def inputs(batch):''')
replace('    for m in modules: m.gradient_compressor = comp\n',
        '    for m in modules: m.gradient_compressor = comp\n    DIAG.install(model,modules,sch)\n')
replace("            for i in range(plan['steps_per_method']):",f'            for i in range({a.max_steps}):\n                DIAG.start_step(i+1)')
replace('                result = model(**inputs(next(iterator)))',
        '                data = next(iterator)\n                DIAG.batch(data)\n                result = model(**inputs(data))')
replace('                loss.backward()', '                DIAG.loss(loss)\n                loss.backward()')
replace('                completed = i+1', '                completed = i+1\n                DIAG.end_step(completed,value)')
replace("        if plan['reporting']['save_final_weights']:","        if False:  # bounded diagnostic: no final weights or long-run artifacts")
replace("        summary['training_seconds'] = sum(x['seconds'] for x in summary['training_intervals'])",
        "        summary['training_seconds'] = time.perf_counter()-interval_start  # diagnostic prefix only")
(out/'instrumented_runner.py').write_text(code)
(out/'diagnostic_config.json').write_text(json.dumps(dict(variant=a.variant,trace_from=a.trace_from,max_steps=a.max_steps,
    original_plan_unchanged=True,scope='bounded replay, not long training or a performance measurement',
    instrumentation='full tensor ranges/finiteness from trace_from, initial weights, batch IDs/RNG, failure-only post-event inspection',
    caveat='Observers change allocation and timing; exact failure iteration is not guaranteed for a race'),indent=2)+'\n')
sys.argv=['run_main.py','--campaign-dir',str(out),'--method','zero']
status=1
try:
    exec(compile(code,str(out/'instrumented_runner.py'),'exec'),{'__name__':'__main__','DIAG':trace})
except SystemExit as exc:
    status=exc.code
finally: trace.close()
raise SystemExit(status)
