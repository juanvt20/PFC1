"""Compare working-tree bytes and Git objects to the pinned manifest (read-only)."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
repo = root / 'original' / 'upstream'
manifest = json.loads((root / 'original' / 'UPSTREAM.json').read_text(encoding='utf-8'))
def git(*args):
    return subprocess.check_output(['git', '-C', str(repo), *args])
report = dict(commit=git('rev-parse', 'HEAD').decode().strip(),
              expected_commit=manifest['commit'], files=len(manifest['files']),
              byte_mismatches=[], object_mismatches=[], eol_only=[])
for item in manifest['files']:
    data = (repo / item['path']).read_bytes()
    original = git('cat-file', 'blob', item['git_blob_sha'])
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != item['git_blob_sha']:
        report['byte_mismatches'].append(item['path'])
        if data.replace(b'\r\n', b'\n') == original.replace(b'\r\n', b'\n'):
            report['eol_only'].append(item['path'])
    if git('rev-parse', 'HEAD:' + item['path']).decode().strip() != item['git_blob_sha']:
        report['object_mismatches'].append(item['path'])
report['git_diff_content_empty'] = subprocess.run(
    ['git', '-C', str(repo), '-c', 'core.autocrlf=false', 'diff', '--quiet']).returncode == 0
print(json.dumps(report, indent=2))
raise SystemExit(bool(report['byte_mismatches'] or report['object_mismatches'] or
                      report['commit'] != report['expected_commit']))
