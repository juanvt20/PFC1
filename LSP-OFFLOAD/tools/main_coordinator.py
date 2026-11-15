"""Detached container coordinator; one attempt per stage, never an implicit resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument('--campaign-dir', required=True)
a = p.parse_args()
campaign = Path(a.campaign_dir)
plan = json.loads((campaign/'plan.json').read_text())
started = time.time()
status = dict(campaign=plan['campaign'], state='starting', coordinator_pid=os.getpid(),
              started_unix=started, completed=[], exact_resume_supported=False)


def persist():
    status['updated_unix'] = time.time()
    status['elapsed_seconds'] = time.time()-started
    temporary = campaign/'status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2)+'\n')
    os.replace(temporary, campaign/'status.json')


def memory_events():
    path = Path('/sys/fs/cgroup/memory.events')
    return path.read_text() if path.exists() else None


code = 0
try:
    # An exclusive marker prevents a second coordinator even within the same container.
    with (campaign/'coordinator.started').open('x') as file:
        file.write(str(os.getpid()))
    launch = json.loads((campaign/'launch.json').read_text(encoding='utf-8-sig'))
    for name, expected in launch['script_hashes'].items():
        assert hashlib.sha256((campaign/'code'/name).read_bytes()).hexdigest() == expected
    reused = json.loads((campaign/'reused-short-comparison.json').read_text())
    assert reused['pilot_gate']['passed']
    assert all(reused['performance'][m]['repetitions'] == 3 for m in ['zero','lsp'])
    persist()
    for method in plan['method_order']:
        if (campaign/method).exists():
            raise RuntimeError('Stage output already exists; refusing overwrite or unvalidated resume: '+method)
        argv = [sys.executable, '-u', str(campaign/'code/run_main.py'),
                '--campaign-dir', str(campaign), '--method', method]
        status.update(state='running', stage=method, argv=argv)
        before = memory_events()
        stage_start = time.time()
        print('Starting', method, flush=True)
        with (campaign/(method+'.log')).open('x') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = process.pid; persist()
            while True:
                try:
                    process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    persist()
        elapsed = time.time()-stage_start
        record = dict(argv=argv, exit_code=process.returncode, started_unix=stage_start,
                      ended_unix=time.time(), real_stage_seconds=elapsed,
                      cgroup_memory_events_before=before, cgroup_memory_events_after=memory_events())
        (campaign/(method+'-command.json')).write_text(json.dumps(record, indent=2)+'\n')
        summary_path = campaign/method/'summary.json'
        result = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if process.returncode != 0 or not result or result['status'] != 'success':
            status.update(state='failed', failed_stage=method, failure_record=record,
                error=result.get('error') if result else 'Child ended without summary; inspect log and cgroup memory events')
            code = 1; break
        if method != 'initial':
            assert result['completed_steps'] == plan['steps_per_method']
            assert [x['step'] for x in result['evaluations']] == [1000,2000,3000,4000,5000]
        status['completed'].append(method); persist()
        print('Completed', method, elapsed, 'seconds', flush=True)
    status['state_before_report'] = status['state']
    status['state'] = 'reporting'; persist()
    with (campaign/'report.log').open('x') as log:
        report = subprocess.run([sys.executable, str(campaign/'code/report_main.py'),
            '--campaign-dir', str(campaign)], stdout=log, stderr=subprocess.STDOUT)
    status['report_exit_code'] = report.returncode
    code = code or report.returncode
    status['state'] = 'complete' if code == 0 else 'failed'
except BaseException:
    code = 1
    status.update(state='failed', coordinator_error=traceback.format_exc())
    print(status['coordinator_error'], flush=True)
finally:
    status['exit_code'] = code
    persist()
    print(json.dumps(status, indent=2), flush=True)
raise SystemExit(code)
