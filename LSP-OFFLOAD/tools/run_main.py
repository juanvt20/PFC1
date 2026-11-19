"""Authorized 5000-step campaign. Algorithm modules are the validated, unchanged copy."""
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

import numpy as np
import psutil
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding

p = argparse.ArgumentParser()
p.add_argument('--campaign-dir', required=True)
p.add_argument('--method', choices=['initial', 'zero', 'lsp'], required=True)
a = p.parse_args()
root = Path('/work')
campaign = Path(a.campaign_dir)
plan = json.loads((campaign/'plan.json').read_text())
out = campaign/a.method
out.mkdir(exist_ok=False)
sys.path.insert(0, str(root/'adaptado/upstream'))
from para_offload_modules import replace_linear_layer_with_parallel_linear, Scheduler
from gradient_compressor import GradientCompressor, LearnedCountSketchCompressor
from llm_finetuning import calculate_ppl
from state import State

common = plan['common']
torch.set_num_threads(plan['cpu_threads'])
torch.set_num_interop_threads(plan['interop_threads'])
torch.manual_seed(common['seed'])
started = time.perf_counter()
phase, completed = 'startup', 0
stop = threading.Event()
worker = None
peaks = {}
summary = dict(method=a.method, status='running', completed_steps=0, evaluations=[],
               calibrations=[], training_intervals=[], convergence_claim=False,
               exact_resume_supported=False, final_weights_only=False)


def save(name, value):
    target = out/name
    temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, target)


def progress(**extra):
    save('progress.json', dict(method=a.method, phase=phase, completed_steps=completed,
                              elapsed_seconds=time.perf_counter()-started, **extra))


def snapshot():
    mem = psutil.virtual_memory()
    free, total = torch.cuda.mem_get_info()
    row = dict(elapsed_s=time.perf_counter()-started, phase=phase, step=completed,
               rss_bytes=psutil.Process().memory_info().rss, ram_available_bytes=mem.available,
               ram_total_bytes=mem.total, swap_used_bytes=psutil.swap_memory().used,
               gpu_allocated_bytes=torch.cuda.memory_allocated(), gpu_reserved_bytes=torch.cuda.memory_reserved(),
               gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(),
               gpu_max_reserved_bytes=torch.cuda.max_memory_reserved(), gpu_free_bytes=free, gpu_total_bytes=total)
    for key in ['memory.current', 'memory.peak']:
        path = Path('/sys/fs/cgroup')/key
        row[key] = int(path.read_text()) if path.exists() else None
    return row


def monitor():
    with (out/'samples.csv').open('w', newline='') as file:
        writer = None
        while not stop.is_set():
            try:
                row = snapshot()
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); file.flush()
                for key in ['rss_bytes','gpu_max_allocated_bytes','gpu_max_reserved_bytes','memory.current','memory.peak']:
                    if row[key] is not None:
                        peaks[key] = max(peaks.get(key, 0), row[key])
            except Exception as exc:
                summary['monitor_error'] = repr(exc)
            stop.wait(plan['reporting']['resource_sample_seconds'])


class Alpaca(Dataset):
    def __init__(self, raw, bounds, tokenizer):
        self.raw, self.indices, self.tokenizer = raw, range(*bounds), tokenizer
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        item = self.raw[self.indices[index]]
        text = f"instruction: {item['instruction']}\n\ninput: {item['input']}\n\noutput: {item['output']}"
        return self.tokenizer(text, padding=False, max_length=common['max_sequence_length'], truncation=True)


def inputs(batch):
    batch = {k:v.cuda() for k,v in batch.items()}
    batch['labels'] = batch['input_ids']
    return batch


exit_code = 0
try:
    for relative, expected in plan['immutable_reference_hashes'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative
    torch.cuda.init()
    worker = threading.Thread(target=monitor, daemon=True); worker.start()
    save('environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, transformers=transformers.__version__, numpy=np.__version__,
        device=torch.cuda.get_device_name(), initial=snapshot()))
    save('config.json', dict(plan=plan, method=a.method,
        launch=json.loads((campaign/'launch.json').read_text(encoding='utf-8-sig'))))
    phase = 'data'; progress()
    raw = json.loads((root/plan['dataset_asset']['path']).read_text())
    assert len(raw) == plan['dataset_asset']['examples']
    model_path = str(root/plan['model_asset']['snapshot'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, padding='longest')
    train = DataLoader(Alpaca(raw, plan['split']['train'], tokenizer), batch_size=common['batch'],
        shuffle=True, collate_fn=collator, generator=torch.Generator().manual_seed(common['seed']))
    dev = DataLoader(Alpaca(raw, plan['split']['calibration'], tokenizer), batch_size=common['batch'], collate_fn=collator)
    evaluation = DataLoader(Alpaca(raw, plan['split']['evaluation'], tokenizer), batch_size=common['batch'], collate_fn=collator)
    assert len(evaluation.dataset) == plan['evaluation_examples'] == 5185
    phase = 'model_load'; progress()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    for param in model.parameters(): param.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    state = {'state': State.RUNNING}
    modules = replace_linear_layer_with_parallel_linear(model, modules=plan['modules'],
        use_lora=False, lora_rank=None, offload='cpu', lr=common['learning_rate'],
        optim_dtype=torch.float32, use_grad_ckpt=True, global_state=state)
    state['offload_modules'] = modules
    assert len(modules) == plan['expected_offload_matrices']
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == plan['expected_trainable_parameters']
    assert all(m.optim.eps == common['epsilon'] for m in modules)
    route = plan['methods']['lsp' if a.method == 'lsp' else 'zero']
    sch = Scheduler(modules, **route['scheduler'])
    comp = (LearnedCountSketchCompressor(sketch_size=route['d'], n_unempty=route['r'],
        n_iter=route['n_iter'], n_ft_iter=route['n_ft_iter'], lr=route['lr'], ft_lr=route['ft_lr'],
        reuse_sketch=route['reuse_sketch'], cs_init=route['cs_init'], thresh_hold=route['thresh_hold'])
        if a.method == 'lsp' else GradientCompressor())
    for m in modules: m.gradient_compressor = comp

    def evaluate(step):
        global phase
        phase = 'evaluation'; sch.drain(); model.eval(); progress(evaluation_step=step)
        before = [m.optim.t for m in modules]
        t = time.perf_counter()
        values = []
        with (out/f'eval-batches-{step:05d}.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['batch','loss','elapsed_seconds']); writer.writeheader()
            def capture(_model, _inputs, outputs):
                value = outputs.loss.mean().item()
                values.append(value)
                writer.writerow(dict(batch=len(values), loss=value, elapsed_seconds=time.perf_counter()-t)); file.flush()
                if len(values) % 100 == 0: progress(evaluation_step=step, evaluation_batches=len(values))
            handle = model.register_forward_hook(capture)
            try:
                ppl = float(calculate_ppl(model, evaluation, max_len=common['max_sequence_length'], prob=1.0))
            finally:
                handle.remove()
        torch.cuda.synchronize()
        assert math.isfinite(ppl) and len(values) == len(evaluation)
        assert ppl == float(np.exp(np.mean(values)))
        assert before == [m.optim.t for m in modules]
        row = dict(step=step, mean_batch_loss=float(np.mean(values)), perplexity=ppl,
            examples=len(evaluation.dataset), batches=len(values), seconds=time.perf_counter()-t,
            elapsed_seconds=time.perf_counter()-started, shared_initial=step == 0,
            final=step == plan['steps_per_method'])
        summary['evaluations'].append(row)
        with (out/'evaluations.csv').open('a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if file.tell() == 0: writer.writeheader()
            writer.writerow(row); file.flush()
        save('summary.json', summary); progress(evaluation=row)
        print('Evaluation', json.dumps(row), flush=True)

    def calibrate(step):
        global phase
        phase = 'calibration'; sch.drain(); progress(calibration_step=step)
        t = time.perf_counter()
        before = next(iter(comp.sketches_map.values()))[0].values.detach().clone() if comp.sketches_map else None
        counts = [m.optim.t for m in modules]
        for m in modules: m.to('cpu')
        state['state'] = State.RUNNING_PROFILE
        model.train()
        for batch in dev: model(**inputs(batch)).loss.backward()
        model.cpu(); comp.fit(modules); model.cuda()
        state['state'] = State.RUNNING
        torch.cuda.synchronize()
        assert counts == [m.optim.t for m in modules]
        if any(m.optim.m.device.type != 'cpu' for m in modules):
            raise RuntimeError('Calibration left CPU optimizer moments on another device; no automatic formula/device patch')
        after = next(iter(comp.sketches_map.values()))[0].values.detach()
        row = dict(step=step, seconds=time.perf_counter()-t, initialization=before is None,
            learned_recalibration=before is not None and not torch.equal(before, after),
            elapsed_seconds=time.perf_counter()-started)
        summary['calibrations'].append(row)
        save('summary.json', summary); progress(calibration=row)
        print('Calibration', json.dumps(row), flush=True)

    if a.method == 'initial':
        evaluate(0)
    else:
        initial_path = campaign/'initial/summary.json'
        initial = json.loads(initial_path.read_text())
        assert initial['status'] == 'success' and initial['evaluations'][0]['examples'] == 5185
        summary['common_initial'] = dict(path='../initial/summary.json',
            sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(), evaluation=initial['evaluations'][0])
        phase = 'optimizer_init'; progress()
        if a.method == 'lsp': calibrate(0)
        else: comp.fit(modules)
        torch.manual_seed(common['seed'])
        iterator = iter(train)
        phase = 'training'; progress()
        interval_start = time.perf_counter()
        interval_first = 1
        with (out/'steps.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['step','loss','step_seconds','elapsed_seconds','warmup_tag']); writer.writeheader()
            for i in range(plan['steps_per_method']):
                if a.method == 'lsp' and i > 0 and i % route['check_frequency'] == 0:
                    calibrate(i)
                    phase = 'training'; progress(); interval_start = time.perf_counter()
                t = time.perf_counter()
                model.train()
                result = model(**inputs(next(iterator)))
                regularization = sum(torch.norm(param) for param in model.parameters())
                loss = result.loss + common['l2_reg'] * regularization
                loss.backward()
                value = loss.item()
                if not math.isfinite(value): raise RuntimeError('Non-finite training loss')
                completed = i+1
                row = dict(step=completed, loss=value, step_seconds=time.perf_counter()-t,
                    elapsed_seconds=time.perf_counter()-started, warmup_tag=completed <= 5)
                writer.writerow(row); file.flush()
                if completed % plan['reporting']['progress_every_steps'] == 0:
                    summary.update(completed_steps=completed, last_train_loss=value)
                    save('summary.json', summary); progress(last_step=row)
                    print('Step', completed, 'loss', value, 'elapsed_s', row['elapsed_seconds'], flush=True)
                if completed % plan['evaluation_frequency'] == 0 or completed == plan['steps_per_method']:
                    drain_start = time.perf_counter(); sch.drain()
                    drain_seconds = time.perf_counter()-drain_start
                    assert [m.optim.t for m in modules] == [completed]*len(modules)
                    duration = time.perf_counter()-interval_start
                    count = completed-interval_first+1
                    summary['training_intervals'].append(dict(first_step=interval_first, last_step=completed,
                        iterations=count, seconds=duration, seconds_per_iteration=duration/count,
                        iterations_per_second=count/duration, final_drain_seconds=drain_seconds,
                        includes_first_five_steps=interval_first == 1))
                    evaluate(completed)
                    phase = 'training'; progress(); interval_first = completed+1; interval_start = time.perf_counter()
        phase = 'finish'; sch.drain()
        summary['adam_counters'] = [m.optim.t for m in modules]
        summary['training_seconds'] = sum(x['seconds'] for x in summary['training_intervals'])
        summary['iterations_per_second'] = completed/summary['training_seconds']
        summary['seconds_per_iteration'] = summary['training_seconds']/completed
        if plan['reporting']['save_final_weights']:
            phase = 'save_weights'; progress()
            t = time.perf_counter(); sch.drain()
            torch.save(model.state_dict(), out/'final-weights-only.pt')
            summary.update(final_weights_only=True, weight_save_seconds=time.perf_counter()-t)
            save('weight-artifact.json', dict(path='final-weights-only.pt', step=completed,
                purpose='final model weights for evaluation; NOT an exact training-resume checkpoint',
                missing_state=['CPU Adam moments and counters','projectors','RNG','data cursor','scheduler state']))
    summary['status'] = 'success'
except BaseException as exc:
    message = str(exc).lower()
    oom = isinstance(exc, torch.OutOfMemoryError) or any(s in message for s in ['out of memory','cannot allocate memory'])
    exit_code = 2 if oom else 1
    summary.update(status='oom' if oom else 'failed', failure_phase=phase, error=traceback.format_exc())
    print(summary['error'], flush=True)
finally:
    stop.set()
    if worker: worker.join(timeout=5)
    summary.update(completed_steps=completed, exit_code=exit_code, memory_peaks=peaks,
                   real_total_seconds=time.perf_counter()-started)
    save('summary.json', summary); progress(status=summary['status'])
    print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(exit_code)
