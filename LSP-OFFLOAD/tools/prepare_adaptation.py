"""Rebuild an ignored copy and the separate authorized patches from raw blobs."""
import difflib
import hashlib
import json
from pathlib import Path
from scheduler_patch import apply_scheduler_patch

root = Path(__file__).resolve().parents[1]
source = root / 'original/upstream'
target = root / 'adaptado/upstream'
manifest = json.loads((root / 'original/UPSTREAM.json').read_text(encoding='utf-8'))
originals = {}
for item in manifest['files']:
    raw = (source / item['path']).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != item['git_blob_sha']:
        raise RuntimeError('Original bytes differ: ' + item['path'])
    originals[item['path']] = raw

modified = dict(originals)
filename = 'gradient_compressor.py'
old = b'                module.init(self.sketch_size, self.sketch_size)'
new = (b'                module.init(self.sketch_size, self.sketch_size,\n'
       b'                            preserve_optimizer_state=module.inited)')
assert modified[filename].count(old) == 1
modified[filename] = modified[filename].replace(old, new)

filename = 'para_offload_modules.py'
old = (b'    def init(self, left_compress_size, right_compress_size):\n'
       b'        self.inited = True\n'
       b'        self.optim.init((left_compress_size, right_compress_size))')
new = (b'    def init(self, left_compress_size, right_compress_size, preserve_optimizer_state=False):\n'
       b'        if preserve_optimizer_state:\n'
       b'            assert self.inited\n'
       b'            shape = (left_compress_size, right_compress_size)\n'
       b'            assert self.optim.m.shape == self.optim.v.shape == shape\n'
       b'            # fit has already transported M/V; only restore their execution device.\n'
       b'            self.optim.m = self.optim.m.to(self.offloading_device).contiguous()\n'
       b'            self.optim.v = self.optim.v.to(self.offloading_device).contiguous()\n'
       b'            self.current_device = self.offloading_device\n'
       b'        else:\n'
       b'            self.optim.init((left_compress_size, right_compress_size))\n'
       b'        self.inited = True')
assert modified[filename].count(old) == 1
modified[filename] = modified[filename].replace(old, new)

stage_one = dict(modified)
modified = apply_scheduler_patch(stage_one)

# Only upgrade recognized patch-1 bytes, never overwrite unrelated local edits.
if target.exists():
    for path, raw in modified.items():
        dst = target / path
        if dst.exists() and dst.read_bytes() not in (raw, stage_one[path]):
            raise RuntimeError('Existing adapted file differs; inspect before continuing: ' + path)

changed = []
for path, raw in modified.items():
    dst = target / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_bytes() != raw:
        dst.write_bytes(raw)
    if raw != originals[path]:
        changed.append(path)
patchdir = root / 'adaptado/patches'
patchdir.mkdir(exist_ok=True)
for patchname, before, after in [
        ('0001-preserve-transported-moments.patch', originals, stage_one),
        ('0002-drain-pending-operations.patch', stage_one, modified)]:
    patch = ''.join(''.join(difflib.unified_diff(before[p].decode('utf-8').splitlines(keepends=True),
                  after[p].decode('utf-8').splitlines(keepends=True), fromfile='a/'+p, tofile='b/'+p))
                  for p in before if before[p] != after[p])
    (patchdir / patchname).write_bytes(patch.encode('utf-8'))
print(json.dumps(dict(verified_original_files=len(originals), changed_files=changed,
                     patches=sorted(p.name for p in patchdir.glob('*.patch')))))
