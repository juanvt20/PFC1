"""Summarize measured runs; cost extrapolations are separate from measurements."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
plan = json.loads((root/'configs/short_comparison.json').read_text())


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


report = dict(campaign=plan['campaign'], convergence_claim=False, runs=[],
              performance={}, main_training_estimate=None)
for path in sorted((root/'results').glob(plan['campaign']+'-*/summary.json')):
    summary = read(path)
    command = read(path.parent/'command.json')
    row = dict(summary=summary, command=command, local_path=str(path.parent.relative_to(root)),
               summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config_path = path.parent/'config.json'
    if config_path.exists():
        row['config'] = read(config_path)
    sample_path = path.parent/'samples.csv'
    if sample_path.exists():
        with sample_path.open() as file:
            samples = list(csv.DictReader(file))
        if samples:
            row['last_sample'] = samples[-1]
            row['sampled_peak_rss_bytes'] = max(int(x['rss_bytes']) for x in samples)
    report['runs'].append(row)

gate_path = root/'results'/f"{plan['campaign']}-pilot-gate.json"
if gate_path.exists():
    gate = read(gate_path)
    report['pilot_gate'] = gate
    for method in ['zero','lsp']:
        runs = [r for r in report['runs'] if r['summary']['model']==gate['model']
                and r['summary']['method']==method and r['summary']['kind']=='performance'
                and r['summary']['status']=='success']
        times = [r['summary']['performance_window']['seconds_per_iteration'] for r in runs]
        rates = [r['summary']['performance_window']['iterations_per_second'] for r in runs]
        if times:
            report['performance'][method] = dict(repetitions=len(times),
                seconds_per_iteration=times, iterations_per_second=rates,
                mean_seconds_per_iteration=statistics.mean(times),
                mean_iterations_per_second=statistics.mean(rates),
                std_seconds_per_iteration=statistics.stdev(times) if len(times)>1 else None,
                std_definition='sample std across independent short runs, not within-curve fluctuations',
                recalibration_iterations_in_windows=0)
    if all(report['performance'].get(m,{}).get('repetitions')==3 for m in ['zero','lsp']):
        estimate = dict(model=gate['model'],steps=5000,batch=4,max_sequence_length=512,
            check_frequency=1000,evaluation_frequency=1000,methods={},
            nature='extrapolation from measured short runs, not an observed long training',
            limitations=['Calibration measured at pilot step50 may differ later',
                        'Full evaluation time extrapolates 64-example diagnostic time by example count',
                        'Laptop thermals, sequence lengths, memory pressure and projector checks may alter cost'])
        for method in ['zero','lsp']:
            pilot = next(r for r in report['runs'] if r['summary']['run_id']==gate[method])
            s,c = pilot['summary'],pilot['config']
            full_eval = c['split']['original_eval'][1]-c['split']['original_eval'][0]
            cal_initial = sum(x['seconds'] for x in s['calibrations'] if x['initialization'])
            recal = [x['seconds'] for x in s['calibrations'] if not x['initialization']]
            cal_later = statistics.mean(recal) if recal else 0.0
            full_eval_seconds = s['evaluations'][0]['seconds']*full_eval/s['evaluations'][0]['examples']
            startup = max(0.0,pilot['command']['host_wall_seconds']-s['train_seconds']
                          -sum(x['seconds'] for x in s['calibrations'])-sum(x['seconds'] for x in s['evaluations']))
            total = startup+5000*report['performance'][method]['mean_seconds_per_iteration']+cal_initial+4*cal_later+5*full_eval_seconds
            estimate['methods'][method] = dict(estimated_seconds=total,estimated_hours=total/3600,
                startup_seconds=startup,initial_calibration_seconds=cal_initial,
                measured_recalibration_seconds=cal_later,estimated_full_evaluation_seconds=full_eval_seconds,
                full_evaluation_examples=full_eval)
        estimate['combined_estimated_hours']=sum(x['estimated_hours'] for x in estimate['methods'].values())
        estimate['proposed_budget_hours_with_50pct_margin']=1.5*estimate['combined_estimated_hours']
        report['main_training_estimate']=estimate

target=root/'configs/comparison_summary.json'
target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(runs=len(report['runs']),performance=report['performance'],
                     main_training_estimate=report['main_training_estimate']),indent=2))
