"""Exercise Hugging Face GPT-2 reentrant checkpointing, without downloading weights."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import torch
from transformers import GPT2Config, GPT2LMHeadModel

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--run-id',required=True)
a=p.parse_args()
root=Path('/work')
out=root/'results'/a.run_id
out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/a.source/'upstream'))
from para_offload_modules import replace_linear_layer_with_parallel_linear, Scheduler, ScheduleEnv
from gradient_compressor import GradientCompressor, LearnedCountSketchCompressor
from state import State
torch.set_num_threads(2)
report={'scope':'tiny_random_GPT2_checkpointing_not_pretrained_pilot','cases':[]}


def run(route, input_grad_hook):
    torch.manual_seed(2026)
    model=GPT2LMHeadModel(GPT2Config(n_layer=2,n_head=2,n_embd=32,vocab_size=128,
        n_positions=32,use_cache=False,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)).to('cuda',torch.bfloat16)
    for param in model.parameters(): param.requires_grad_(False)
    model.gradient_checkpointing_enable()
    if input_grad_hook:
        model.enable_input_require_grads() # Activations only; embedding weights stay frozen.
    state={'state':State.RUNNING}
    modules=replace_linear_layer_with_parallel_linear(model,modules=['attn.qkv','attn.o','mlp.up','mlp.down'],
        use_lora=False,lora_rank=None,offload='cpu',lr=1e-4,optim_dtype=torch.float32,use_grad_ckpt=True,global_state=state)
    comp=LearnedCountSketchCompressor(sketch_size=16,n_unempty=1,cs_init='gaussian') if route=='lsp' else GradientCompressor()
    comp.fit(modules)
    for m in modules: m.gradient_compressor=comp
    sch=Scheduler(modules,zero=route=='zero',fcfs_point=4,fcfs_process_delay=1,
        fcfs_h2d_delay=1,fcfs_upd_delay=1,lcfs_d2h_delay=1000,lcfs_process_delay=1,lcfs_h2d_delay=0)
    losses=[]
    with ScheduleEnv(sch):
        for step in range(6):
            model.train()
            tokens=torch.randint(0,128,(2,16),device='cuda')
            loss=model(tokens,labels=tokens).loss
            loss.backward()
            losses.append(loss.item())
            if step==2:
                sch.drain()
                model.eval()
                with torch.no_grad():
                    assert torch.isfinite(model(tokens,labels=tokens).loss)
    return dict(passed=all(m.optim.t==6 for m in modules) and not sch.pending_work(),
                modules=len(modules),adam_steps=[m.optim.t for m in modules],losses=losses,
                embedding_weights_trainable=model.get_input_embeddings().weight.requires_grad)


for route,hook in [('zero',False),('zero',True),('lsp',True)]:
    try:
        row=run(route,hook)
    except Exception:
        row={'passed':False,'error':traceback.format_exc()}
    row.update(route=route,input_grad_hook=hook)
    report['cases'].append(row)
report['passed']=all(c['passed'] for c in report['cases'] if c['input_grad_hook'])
(out/'diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
raise SystemExit(not report['passed'])
