"""Extract page text and render selected pages; never modify source PDFs."""
import argparse
import hashlib
import json
from pathlib import Path
import pymupdf

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--pages', default='')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / 'referencias' / args.name
out = root / 'tools' / 'pdf_output' / source.stem
out.mkdir(parents=True, exist_ok=True)
doc = pymupdf.open(source)
meta = dict(file=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            pages=len(doc), pymupdf=pymupdf.VersionBind, extraction=[])
for i, page in enumerate(doc, 1):
    txt = page.get_text(sort=True)
    (out / f'page-{i:03}.txt').write_text(txt, encoding='utf-8')
    meta['extraction'].append(dict(page=i, chars=len(txt)))
selected = range(1, len(doc)+1) if args.pages == 'all' else map(int, filter(None, args.pages.split(',')))
for i in selected:
    doc[i-1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(out / f'page-{i:03}.png')
(out / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(json.dumps(meta, indent=2))
