"""Verify reuse of a separate, full initial evaluation without touching training RNG."""
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reuse_initial(root, campaign, plan):
    spec = plan['initial_evaluation_reuse']
    source = root / spec['source_campaign']
    assert source.resolve().is_relative_to((root / 'results').resolve())
    assert source != campaign
    assert sha256(source / 'plan.json') == spec['source_plan_sha256']
    previous = json.loads((source / 'plan.json').read_text())
    metadata = {'config_version', 'campaign', 'code_base_commit', 'reuse', 'estimate',
                'differences_from_paper', 'immutable_reference_hashes',
                'initial_evaluation_reuse', 'integration'}
    assert {k: v for k, v in plan.items() if k not in metadata} == {
        k: v for k, v in previous.items() if k not in metadata}, 'Experimental configuration differs'
    assert sha256(campaign / 'code/run_main.py') == spec['runner_sha256']
    assert sha256(source / 'code/run_main.py') == spec['runner_sha256']
    saved_config = json.loads((source / 'initial/config.json').read_text())
    assert saved_config['method'] == 'initial' and saved_config['plan'] == previous
    summary = source / 'initial/summary.json'
    assert sha256(summary) == spec['summary_sha256']
    result = json.loads(summary.read_text())
    assert result['status'] == 'success' and result['completed_steps'] == 0
    assert len(result['evaluations']) == 1
    evaluation = result['evaluations'][0]
    assert evaluation['step'] == 0 and evaluation['examples'] == 5185 and evaluation['batches'] == 1297
    assets = {}
    for name, expected in plan['model_asset']['files'].items():
        path = root / plan['model_asset']['snapshot'] / name
        assert path.stat().st_size == expected['bytes']
        assets[str(path.relative_to(root))] = sha256(path)
        assert assets[str(path.relative_to(root))] == expected['sha256'], name
    dataset = root / plan['dataset_asset']['path']
    assets[plan['dataset_asset']['path']] = sha256(dataset)
    assert assets[plan['dataset_asset']['path']] == plan['dataset_asset']['sha256']
    target = campaign / 'initial'
    target.mkdir(exist_ok=False)
    copied = {}
    for name in ['summary.json', 'eval-batches-00000.csv', 'evaluations.csv']:
        shutil.copyfile(source / 'initial' / name, target / name)
        copied[name] = sha256(target / name)
        assert copied[name] == sha256(source / 'initial' / name)
    record = dict(reused=True, source_campaign=spec['source_campaign'],
        source_config_version=previous['config_version'], source_summary_sha256=spec['summary_sha256'],
        verified_assets=assets, copied_files=copied, evaluation= evaluation,
        original_evaluation_seconds=evaluation['seconds'], evaluation_seconds_in_this_campaign=0,
        training_rng_unchanged=True, rng_basis=spec['rng_basis'],
        no_training_state_loaded=True, runner_sha256=spec['runner_sha256'])
    (target / 'reuse.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
