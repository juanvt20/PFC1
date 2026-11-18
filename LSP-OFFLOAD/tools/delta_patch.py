"""Exact authorized patch 0003; no optimizer or scheduler formula changes."""
BEFORE = b'''                assert torch.isnan(delta).sum().item() == 0
                torch.cuda.current_stream().wait_event(event)
                delta.record_stream(torch.cuda.current_stream())
'''
AFTER = b'''                torch.cuda.current_stream().wait_event(event)
                delta.record_stream(torch.cuda.current_stream())
                assert torch.isfinite(delta).all().item()
'''


def apply_delta_patch(source):
    assert source.count(BEFORE) == 1
    assert source.count(AFTER) == 0
    return source.replace(BEFORE, AFTER)
