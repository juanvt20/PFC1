"""Short real GPT-2/Alpaca runs; original CPU optimizer and explicit scheduler routes."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback

import psutil
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding

p=argparse.ArgumentParser()
p.add_argument('--model',choices=['gpt2-large','gpt2-medium'],required=True)
p.add_argument('--method',choices=['zero','lsp'],required=True)
p.add_argument('--kind',choices=['pilot','performance'],required=True)
p.add_argument('--run-id',required=True)
p.add_argument('--steps',type=int,default=100)
p.add_argument('--warmup',type=int,default=5)
p.add_argument('--seed',type=int,default=2026)
a=p.parse_args()
root=Path('/work')
if Path(a.run_id).name!=a.run_id: p.error('invalid run ID')
out=root/'results'/a.run_id
out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/'adaptado/upstream'))
from para_offload_modules import replace_linear_layer_with_parallel_linear, Scheduler
from gradient_compressor import GradientCompressor, LearnedCountSketchCompressor
from state import State
from llm_finetuning import calculate_ppl

torch.set_num_threads(14)
torch.set_num_interop_threads(2)
torch.manual_seed(a.seed)
assets=json.loads((root/'configs/pilot_assets.json').read_text())
model_info=assets['models'][a.model]
dataset_info=assets['dataset']
started=time.perf_counter()
phase='startup'
stop=threading.Event()
samples=[]
summary=dict(run_id=a.run_id,kind=a.kind,status='running',method=a.method,model=a.model,
             convergence_claim=False,performance_window=None,calibrations=[],evaluations=[])
config=dict(**vars(a),batch=4,max_sequence_length=512,weights_dtype='bfloat16',optimizer_dtype='float32',
    optimizer='upstream CPU fused Adam',epsilon=1e-4,learning_rate=1e-4,l2_reg=0.0,
    d=512 if a.method=='lsp' else None,r=8 if a.method=='lsp' else None,
    check_frequency=50 if a.kind=='pilot' else 1000,projector_threshold=.8,
    projector_init='gaussian',projector_iterations=100,projector_specific_iterations=0,
    projector_lr=.1,checkpointing=True,input_activation_grad_hook=True,
    seed_origin='local explicit seed, not claimed as authors seed',
    evaluation='first 64 examples of the original evaluation split; diagnostic subset, not full quality evaluation',
    perplexity_definition='unchanged upstream calculate_ppl: exp(mean(batch loss)), padding/prompt semantics retained',
    model_asset=model_info,dataset_asset=dataset_info,
    git_commit=os.environ.get('PFC_COMMIT'),git_dirty=os.environ.get('PFC_DIRTY'),
    adaptations=['explicit baseline scheduler flag','LSP explicit FCFS/LCFS delays',
        'boundary drain and pre-matmul update dependency','CUDA buffer record_stream',
        'preserve transported moments','HF input activation grad hook with embeddings frozen',
        'short diagnostic evaluation subset64','lazy tokenization without upstream diskcache'],
    versions=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                  transformers=transformers.__version__))
if a.kind=='pilot': config['adaptations'].append('CheckFreq shortened from1000 to50 for pilot only')
if a.model=='gpt2-medium': config['adaptations'].append('single authorized size fallback from GPT2-774M to GPT2-Medium')
config['patches']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'adaptado/patches').glob('*.patch')}


def save_json(name,value):
    (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def resource_row():
    memory=psutil.virtual_memory()
    row=dict(elapsed_s=time.perf_counter()-started,phase=phase,rss_bytes=psutil.Process().memory_info().rss,
             ram_available_bytes=memory.available,ram_total_bytes=memory.total,
             swap_used_bytes=psutil.swap_memory().used,
             gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_reserved_bytes=torch.cuda.memory_reserved(),
             gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_max_reserved_bytes=torch.cuda.max_memory_reserved())
    free,total=torch.cuda.mem_get_info()
    row.update(gpu_free_bytes=free,gpu_total_bytes=total)
    for field in ['memory.current','memory.peak']:
        path=Path('/sys/fs/cgroup')/field
        row[field]=int(path.read_text()) if path.exists() else None
    return row


def monitor():
    with (out/'samples.csv').open('w',newline='') as file:
        writer=None
        while not stop.is_set():
            try:
                row=resource_row()
                if writer is None:
                    writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);file.flush();samples.append(row)
            except Exception as exc:
                summary['monitor_error']=repr(exc)
            stop.wait(1)


class Alpaca(Dataset):
    def __init__(self,raw,indices,tokenizer):
        self.raw,self.indices,self.tokenizer=raw,list(indices),tokenizer
    def __len__(self): return len(self.indices)
    def __getitem__(self,index):
        item=self.raw[self.indices[index]]
        text=f"instruction: {item['instruction']}\n\ninput: {item['input']}\n\noutput: {item['output']}"
        return self.tokenizer(text,padding=False,max_length=512,truncation=True)


def inputs(batch):
    batch={k:v.cuda() for k,v in batch.items()}
    batch['labels']=batch['input_ids']
    return batch


worker=None
exit_code=0
try:
    torch.cuda.init()
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    save_json('config.json',config)
    save_json('environment.json',dict(initial=resource_row(),versions=config['versions'],
                                     device=torch.cuda.get_device_name(),threads=torch.get_num_threads(),
                                     interop_threads=torch.get_num_interop_threads()))
    phase='data'
    raw=json.loads((root/dataset_info['path']).read_text())
    ntrain=int(.9*len(raw));ncal=16
    snapshot=str(root/model_info['snapshot'])
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
    tokenizer.pad_token=tokenizer.eos_token
    collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8,padding='longest')
    train=DataLoader(Alpaca(raw,range(ntrain),tokenizer),batch_size=4,shuffle=True,collate_fn=collator,
                     generator=torch.Generator().manual_seed(a.seed))
    dev=DataLoader(Alpaca(raw,range(ntrain,ntrain+ncal),tokenizer),batch_size=4,collate_fn=collator)
    evaluation=DataLoader(Alpaca(raw,range(ntrain+ncal,min(ntrain+ncal+64,len(raw))),tokenizer),batch_size=4,collate_fn=collator)
    config['split']=dict(train=[0,ntrain],calibration=[ntrain,ntrain+ncal],
                         original_eval=[ntrain+ncal,len(raw)],diagnostic_eval=[ntrain+ncal,ntrain+ncal+len(evaluation.dataset)])
    phase='model_load'
    print('Loading',a.model,a.method,flush=True)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.bfloat16)
    model.config.use_cache=False
    model=model.cuda()
    for param in model.parameters(): param.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    state={'state':State.RUNNING}
    modules=replace_linear_layer_with_parallel_linear(model,modules=['attn.qkv','attn.o','mlp.up','mlp.down'],
        use_lora=False,lora_rank=None,offload='cpu',lr=1e-4,optim_dtype=torch.float32,
        use_grad_ckpt=True,global_state=state)
    state['offload_modules']=modules
    config['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    config['total_parameters']=sum(p.numel() for p in model.parameters())
    config['offload_modules']=[dict(name=name,shape=list(layer.weight.shape)) for name,layer in model.named_modules()
                               if hasattr(layer,'optim') and hasattr(layer.optim,'offloading_device')]
    config['scheduler']=dict(zero=a.method=='zero',fcfs_point=min(32,len(modules)//2),
        fcfs_process_delay=1,fcfs_h2d_delay=2,fcfs_upd_delay=2,
        lcfs_d2h_delay=1000,lcfs_process_delay=2,lcfs_h2d_delay=1)
    sch=Scheduler(modules,**config['scheduler'])
    comp=(LearnedCountSketchCompressor(sketch_size=512,n_unempty=8,n_iter=100,n_ft_iter=0,
        lr=.1,ft_lr=.1,cs_init='gaussian',thresh_hold=.8) if a.method=='lsp' else GradientCompressor())
    for module in modules: module.gradient_compressor=comp
    save_json('config.json',config)

    def calibrate(step):
        global phase
        phase='calibration'
        t=time.perf_counter()
        sch.drain()
        before=next(iter(comp.sketches_map.values()))[0].values.detach().clone() if comp.sketches_map else None
        old_steps=[m.optim.t for m in modules]
        for m in modules: m.to('cpu')
        state['state']=State.RUNNING_PROFILE
        model.train()
        for batch in dev:
            model(**inputs(batch)).loss.backward()
        model.cpu()
        comp.fit(modules)
        model.cuda()
        state['state']=State.RUNNING
        torch.cuda.synchronize()
        after=next(iter(comp.sketches_map.values()))[0].values.detach()
        if old_steps!=[m.optim.t for m in modules]: raise RuntimeError('Calibration changed Adam counters')
        if any(m.optim.m.device.type!='cpu' for m in modules):
            raise RuntimeError('Calibration left CPU optimizer moments on another device')
        row=dict(step=step,seconds=time.perf_counter()-t,initialization=before is None,
                 learned_recalibration=before is not None and not torch.equal(before,after))
        summary['calibrations'].append(row)
        save_json('progress.json',dict(step=step,phase=phase,calibrations=summary['calibrations']))
        print('Calibration',row,flush=True)

    phase='optimizer_init'
    if a.method=='lsp': calibrate(0)
    else: comp.fit(modules)
    initial_weight_sample=modules[0].weight.detach().flatten()[:4096].clone()
    torch.manual_seed(a.seed)
    phase='training'
    total_steps=a.steps+(a.warmup if a.kind=='performance' else 0)
    window_start=None
    train_seconds=0.
    with (out/'steps.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['step','loss','seconds','phase','recalibration']);writer.writeheader()
        iterator=iter(train)
        for i in range(total_steps):
            recal=a.method=='lsp' and i>0 and i%config['check_frequency']==0
            if recal: calibrate(i)
            if a.kind=='performance' and i==a.warmup:
                sch.drain();window_start=time.perf_counter()
            phase='warmup' if a.kind=='performance' and i<a.warmup else 'training'
            t=time.perf_counter()
            model.train()
            batch=inputs(next(iterator))
            result=model(**batch)
            # Retain even the original zero-coefficient regularization computation.
            regularization=sum(torch.norm(param) for param in model.parameters())
            loss=result.loss+config['l2_reg']*regularization
            loss.backward()
            value=loss.item()
            if not math.isfinite(value): raise RuntimeError('Non-finite training loss')
            seconds=time.perf_counter()-t;train_seconds+=seconds
            row=dict(step=i+1,loss=value,seconds=seconds,phase=phase,recalibration=recal)
            writer.writerow(row);file.flush()
            save_json('progress.json',dict(**row,calibrations=summary['calibrations']))
            if (i+1)%10==0: print('Step',i+1,'loss',value,'seconds',seconds,flush=True)
        t=time.perf_counter();sch.drain();train_seconds+=time.perf_counter()-t
        if window_start is not None:
            duration=time.perf_counter()-window_start
            summary['performance_window']=dict(iterations=a.steps,seconds=duration,
                iterations_per_second=a.steps/duration,seconds_per_iteration=duration/a.steps,
                warmup_iterations=a.warmup,recalibration_iterations=0)
    counters=[m.optim.t for m in modules]
    if counters!=[total_steps]*len(modules): raise RuntimeError('Lost optimizer updates: '+str(counters))
    summary.update(completed_steps=total_steps,adam_counters=counters,train_seconds=train_seconds,
        initial_weight_sample_max_change=(modules[0].weight.detach().flatten()[:4096]-initial_weight_sample).abs().max().item(),
        final_train_loss=value,diagnostic_iterations_per_second=total_steps/train_seconds,
        diagnostic_seconds_per_iteration=train_seconds/total_steps)
    phase='evaluation'
    sch.drain();model.eval()
    t=time.perf_counter()
    ppl=float(calculate_ppl(model,evaluation,max_len=512,prob=1.0))
    torch.cuda.synchronize()
    if not math.isfinite(ppl): raise RuntimeError('Non-finite perplexity')
    summary['evaluations'].append(dict(step=total_steps,perplexity=ppl,mean_batch_loss=math.log(ppl),
        mean_batch_loss_origin='log of original evaluator perplexity',examples=len(evaluation.dataset),
        seconds=time.perf_counter()-t,scope='diagnostic_subset_not_convergence'))
    phase='finish';sch.drain()
    if a.kind=='pilot' and a.method=='lsp' and not any(x['learned_recalibration'] for x in summary['calibrations']):
        raise RuntimeError('Pilot did not learn new projectors in a real recalibration')
    summary['status']='success'
except BaseException as exc:
    message=str(exc).lower()
    is_oom=isinstance(exc,torch.OutOfMemoryError) or any(s in message for s in ['out of memory','cannot allocate memory','can\'t allocate memory'])
    summary.update(status='oom' if is_oom else 'failed',failure_phase=phase,error=traceback.format_exc())
    exit_code=2 if is_oom else 1
    print(summary['error'],flush=True)
finally:
    stop.set()
    if worker: worker.join(timeout=5)
    summary['real_total_seconds']=time.perf_counter()-started
    summary['exit_code']=exit_code
    if samples:
        summary['memory_peaks']={key:max(row[key] for row in samples if row[key] is not None) for key in
            ['rss_bytes','gpu_allocated_bytes','gpu_reserved_bytes','gpu_max_allocated_bytes','gpu_max_reserved_bytes','memory.current','memory.peak']
            if any(row[key] is not None for row in samples)}
    save_json('summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
raise SystemExit(exit_code)
