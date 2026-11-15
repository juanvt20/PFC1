"""Fetch only the agreed public GPT-2 checkpoint and Alpaca GPT-4, pinned to revisions."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi, snapshot_download

p=argparse.ArgumentParser()
p.add_argument('--model',choices=['gpt2-large','gpt2-medium'],required=True)
a=p.parse_args()
root=Path('/work')
manifest_path=root/'configs/pilot_assets.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'models':{}}

def get_json(url):
    request=urllib.request.Request(url,headers={'User-Agent':'PFC1-LSP-reproduction'})
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)

if 'dataset' not in manifest:
    repo='Instruction-Tuning-with-GPT-4/GPT-4-LLM'
    revision=get_json('https://api.github.com/repos/'+repo+'/commits/main')['sha']
    url=f'https://raw.githubusercontent.com/{repo}/{revision}/data/alpaca_gpt4_data.json'
    path=root/'datasets/alpaca_gpt4_data.json'
    if path.exists():
        raise RuntimeError('Unrecorded dataset exists; inspect before replacing it')
    print('Downloading pinned Alpaca GPT-4',revision,flush=True)
    with urllib.request.urlopen(url,timeout=120) as response:
        raw=response.read()
    data=json.loads(raw)
    path.write_bytes(raw)
    manifest['dataset']={'repository':repo,'revision':revision,'url':url,
        'path':str(path.relative_to(root)),'sha256':hashlib.sha256(raw).hexdigest(),'examples':len(data)}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')

if a.model not in manifest['models']:
    repo='openai-community/'+a.model
    revision=HfApi(token=False).model_info(repo).sha
    print('Downloading pinned checkpoint',repo,revision,flush=True)
    directory=snapshot_download(repo,revision=revision,cache_dir=str(root/'models/hf-cache'),token=False,
        allow_patterns=['config.json','generation_config.json','model.safetensors',
                        'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'],max_workers=2)
    hashes={}
    for file in sorted(Path(directory).iterdir()):
        if file.is_file():
            with file.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            hashes[file.name]={'sha256':digest,'bytes':file.stat().st_size}
    if 'model.safetensors' not in hashes:
        raise RuntimeError('Expected safetensors checkpoint is missing')
    manifest['models'][a.model]={'repository':repo,'revision':revision,
        'snapshot':str(Path(directory).relative_to(root)),'files':hashes}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'model':a.model,'revision':manifest['models'][a.model]['revision'],
                  'dataset_examples':manifest['dataset']['examples']},indent=2),flush=True)
