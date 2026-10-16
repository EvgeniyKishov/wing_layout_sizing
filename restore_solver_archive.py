"""Restore a cleanup-archived file at its original path, verifying its SHA-256."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('path',help='Original workspace-relative filename, with or without .gz')
p.add_argument('--manifest',type=Path,default=ROOT/'storage_cleanup_20261015/verified_archives.json',help='Archive receipt file; use the newer cleanup or candidate native_archive manifest when applicable')
a=p.parse_args();dest=(ROOT/a.path).resolve()
if dest.suffix=='.gz':dest=dest.with_suffix('')
if not dest.is_relative_to(ROOT/'runs'):p.error('Path must be inside runs')
records=json.loads(a.manifest.read_text())
r=next((r for r in records if (ROOT/r['path']).resolve()==dest),None)
if not r:p.error('File not in verified archive manifest')
if dest.exists():p.error('Destination already exists; nothing overwritten')
h=hashlib.sha256();n=0
with gzip.open(ROOT/r['archive'],'rb') as src,dest.open('xb') as out:
    while chunk:=src.read(8*1024*1024):out.write(chunk);h.update(chunk);n+=len(chunk)
if h.hexdigest()!=r['original_sha256'] or n!=r['original_bytes']:
    raise RuntimeError('Restored file failed verification; do not use it')
print(f'Restored and verified {dest}; compressed archive retained.')
