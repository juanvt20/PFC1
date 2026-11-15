"""Authorized scheduler boundary patch; no optimizer/projector formula changes."""


def replace_once(raw, old, new):
    old, new = old.encode(), new.encode()
    assert raw.count(old) == 1, old[:100]
    return raw.replace(old, new)


def protect_stream_lifetimes(raw):
    raw = replace_once(raw,
        '                    pinned_buffer.copy_(device_buffer, non_blocking=state != State.PROFILE_COMM)\n',
        '                    pinned_buffer.copy_(device_buffer, non_blocking=state != State.PROFILE_COMM)\n'
        '                    device_buffer.record_stream(ctx.d2hStream)\n')
    return replace_once(raw,
        '                torch.cuda.current_stream().wait_event(event)\n',
        '                torch.cuda.current_stream().wait_event(event)\n'
        '                delta.record_stream(torch.cuda.current_stream())\n')


def apply_scheduler_patch(files):
    files = dict(files)
    name = 'para_offload_modules.py'
    files[name] = protect_stream_lifetimes(files[name])
    files[name] = replace_once(files[name], '''        output = input @ weight
        if optim is not None:
            ctx.optim = optim
            ctx.optim.forward(input)
            ctx.save_for_backward(weight)
        return output
''', '''        if optim is not None:
            ctx.optim = optim
            # Pending updates must precede the matmul which consumes this weight.
            ctx.optim.forward(input)
        output = input @ weight
        if optim is not None:
            ctx.save_for_backward(weight)
        return output
''')
    files[name] = replace_once(files[name], '''    def clear(self):
        for module in self.modules:
            module.fwd_hooks.clear()
            module.fwd_end_hooks.clear()
            module.bwd_hooks.clear()
            module.bwd_end_hooks.clear()
''', '''    def pending_work(self):
        return [dict(module=m.idx, hooks=sum(len(getattr(m, name)) for name in
                    ('fwd_hooks', 'fwd_end_hooks', 'bwd_hooks', 'bwd_end_hooks')),
                     buffers=len(m.ctx.stack)) for m in self.modules
                if m.fwd_hooks or m.fwd_end_hooks or m.bwd_hooks or m.bwd_end_hooks or m.ctx.stack]

    def clear(self):
        pending = self.pending_work()
        if pending or getattr(self, '_drain_failed', False):
            raise RuntimeError(f'Cannot clear unfinished scheduler work: {pending}')

    def drain(self):
        # A boundary after a complete backward, not an extra training iteration.
        if getattr(self, '_drain_failed', False):
            raise RuntimeError('Cannot retry a partially failed drain')
        if any(m.bwd_hooks or m.bwd_end_hooks for m in self.modules):
            raise RuntimeError('Cannot drain before backward callbacks have completed')
        try:
            # Same layer and hook order as the next forward, without computing it.
            for module in self.modules:
                for name in ('fwd_hooks', 'fwd_end_hooks'):
                    hooks = getattr(module, name)
                    while hooks:
                        hook = hooks.pop(0)
                        hook()
            for device in {m.compute_device for m in self.modules}:
                torch.cuda.synchronize(device)
            self.clear()
        except BaseException:
            self._drain_failed = True
            raise
''')
    files[name] = replace_once(files[name], '    def __exit__(self, *args): \n'
                               '        self.scheduler.clear()\n',
                               '''    def __exit__(self, exc_type, exc_value, traceback):
        # Failed backward/OOM is not a valid boundary: retain the failed work.
        if exc_type is None or (exc_type is SystemExit and exc_value.code in (None, 0)):
            self.scheduler.drain()
        return False
''')
    name = 'llm_finetuning.py'
    files[name] = replace_once(files[name], '                self.scheduler.clear()\n',
                               '                self.scheduler.drain()\n')
    files[name] = replace_once(files[name], '''                    if is_better:
                        torch.save''', '''                    if is_better:
                        self.scheduler.drain()
                        torch.save''')
    files[name] = replace_once(files[name], '''                profile_start_t = time.time()
                for module in self.global_state['offload_modules']:''', '''                profile_start_t = time.time()
                self.scheduler.drain()
                for module in self.global_state['offload_modules']:''')
    return files
