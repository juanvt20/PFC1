"""Local diagnostic observers; no optimizer formula changes or tensor sanitizing."""
import hashlib
import json
from pathlib import Path
import torch


class NonfiniteObserved(RuntimeError):
    pass


def statistics(t):
    t = t.detach()
    finite = torch.isfinite(t)
    all_finite = bool(finite.all().item())
    if all_finite:
        nan = posinf = neginf = 0
        values = t
    else:
        nan = int(torch.isnan(t).sum().item())
        posinf = int(torch.isposinf(t).sum().item())
        neginf = int(torch.isneginf(t).sum().item())
        values = t[finite]
    return dict(shape=list(t.shape), dtype=str(t.dtype), device=str(t.device),
        numel=t.numel(), nan=nan, posinf=posinf, neginf=neginf, finite=all_finite,
        minimum=float(values.min().item()) if values.numel() else None,
        maximum=float(values.max().item()) if values.numel() else None,
        data_ptr=t.data_ptr())


class Trace:
    def __init__(self, directory, start, initial_scan=True):
        self.out = Path(directory)
        self.start = start
        self.initial_scan = initial_scan
        self.step = 0
        self.ids = []
        self.batches = {}
        self.modules = []
        self.failure = None
        self.op = None
        self.losses = []
        self.file = (self.out/'trace.jsonl').open('x')
        self.batchfile = (self.out/'batches.jsonl').open('x')
        self.event_number = 0

    @property
    def enabled(self): return self.step >= self.start

    def emit(self, stage, module=None, origin=None, **values):
        self.event_number += 1
        row = dict(event=self.event_number, stage=stage, execution_step=self.step,
                   module=module.idx if module is not None else None,
                   module_name=getattr(module,'diag_name',None),
                   origin_step=origin, operation_id=f'{origin}:{module.idx}' if module is not None else None,
                   adam_t=module.optim.t if module is not None else None, **values)
        self.file.write(json.dumps(row,allow_nan=False)+'\n');self.file.flush()
        return row

    def inspect(self, stage, tensor, module=None, origin=None):
        row=self.emit(stage,module,origin,tensor=statistics(tensor))
        if not row['tensor']['finite']:
            self.failure = row
            self.save_failure()
            raise NonfiniteObserved(f'{stage}: {row}')
        return row

    def pending(self):
        return [dict(module=m.idx,name=m.diag_name,t=m.optim.t,
                     hooks={name:[dict(kind=getattr(f,'diag_kind',f.__qualname__),
                                      token=getattr(f,'diag_token',None)) for f in getattr(m,name)]
                            for name in ['fwd_hooks','fwd_end_hooks','bwd_hooks','bwd_end_hooks']},
                     buffers=[dict(device=str(t.device),shape=list(t.shape),data_ptr=t.data_ptr(),
                                   event_complete=e.query()) for e,t in m.ctx.stack]) for m in self.modules]

    def save_failure(self, **extra):
        (self.out/'first_failure.json').write_text(json.dumps(dict(first=self.failure,
            pending=self.pending(), batch_current=self.batches.get(self.step),
            batch_previous=self.batches.get(self.step-1), **extra),indent=2,allow_nan=False)+'\n')

    def start_step(self, step):
        self.step=step; self.ids=[]
        if self.enabled:
            torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),
                       self.out/f'rng-before-{step:03d}.pt')

    def sample(self,index): self.ids.append(int(index))

    def batch(self,batch):
        hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in batch.items()}
        row=dict(step=self.step,dataset_indices=list(self.ids),hashes=hashes,
                 shapes={k:list(v.shape) for k,v in batch.items()})
        self.batches[self.step]=row
        self.batchfile.write(json.dumps(row)+'\n'); self.batchfile.flush()
        if self.enabled: torch.save(dict(batch=batch,dataset_indices=self.ids),self.out/f'batch-{self.step:03d}.pt')

    def loss(self,loss):
        if self.enabled: self.inspect('loss_before_backward',loss)

    def end_step(self,step,value):
        self.losses.append(dict(step=step,loss=value))

    def install(self,model,modules,scheduler):
        import para_offload_modules as off
        self.modules=modules
        names={id(layer.optim):name for name,layer in model.named_modules() if hasattr(layer,'optim')}
        for m in modules: m.diag_name=names[id(m)]
        mapping={id(m.ctx):m for m in modules}
        optim_map={id(m.optim):m for m in modules}
        originals={n:getattr(off,n) for n in ['_d2h_comm','_process','_h2d_comm','_update']}
        adam_step=off.Adam.step
        def observed_adam(optim,grad):
            m=optim_map[id(optim)];origin=self.op
            if self.enabled:
                self.inspect('cpu_gradient_before_adam',grad,m,origin)
                self.inspect('M_before_adam',optim.m,m,origin)
                self.inspect('V_before_adam',optim.v,m,origin)
            result=adam_step(optim,grad)
            if self.enabled:
                self.inspect('M_after_adam',optim.m,m,origin)
                self.inspect('V_after_adam',optim.v,m,origin)
                self.inspect('cpu_delta_after_adam',grad,m,origin)
            return result
        off.Adam.step=observed_adam
        for kind in originals:
            def factory(*args,_kind=kind,**kwargs):
                ctx=args[0];m=mapping[id(ctx)];origin=self.step
                if _kind=='_d2h_comm' and self.enabled:
                    self.inspect('gpu_gradient_before_d2h',args[1],m,origin)
                original=originals[_kind](*args,**kwargs)
                def observed():
                    self.op=origin
                    if _kind=='_h2d_comm' and self.enabled:
                        self.inspect('cpu_delta_before_h2d',args[1],m,origin)
                    event=delta=None
                    if _kind=='_update':
                        # Only query readiness; do not read or synchronize the GPU buffer before the original assertion.
                        event,delta=ctx.stack[-1]
                        ready_before=event.query()
                    try:
                        result=original()
                    except BaseException as exc:
                        callback_failure=self.emit('callback_failure',m,origin,callback=_kind,
                            error=repr(exc),event_ready_before=ready_before if event is not None else None)
                        if self.failure is None: self.failure=callback_failure
                        if event is not None:
                            ready_at_exception=event.query()
                            # Failure-only inspection: the assertion has already fired; no update is retried.
                            event.synchronize()
                            settled=delta.detach().cpu()
                            source=m.pinned_buffer_grad
                            details=dict(event_ready_at_exception=ready_at_exception,
                                gpu_delta_after_event=statistics(settled),cpu_delta=statistics(source),
                                delta_matches_cpu_exactly=torch.equal(settled,source),
                                M=statistics(m.optim.m),V=statistics(m.optim.v),weight=statistics(m.weight))
                            self.save_failure(**details)
                        else: self.save_failure()
                        raise
                    if _kind=='_update' and self.enabled:
                        self.inspect('gpu_delta_after_update_wait',delta,m,origin)
                        self.inspect('weight_after_update',args[1],m,origin)
                        self.emit('update_complete',m,origin,event_ready_before=ready_before)
                    return result
                observed.diag_kind=_kind;observed.diag_token=f'{origin}:{m.idx}'
                return observed
            setattr(off,kind,factory)
        if self.initial_scan:
            for m in modules: self.inspect('initial_weight',m.weight,m,0)

    def close(self):
        self.file.close();self.batchfile.close()
        (self.out/'losses.json').write_text(json.dumps(self.losses,indent=2)+'\n')
