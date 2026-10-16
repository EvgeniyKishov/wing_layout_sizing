"""Losslessly archive closed native checkpoints; verified deletion uses PowerShell."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
NATIVE={'.rst','.db','.esav','.mode'}


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def inventory():
    files=[p for p in ROOT.rglob('*') if p.is_file()]
    return dict(bytes=sum(p.stat().st_size for p in files),files=len(files))


def pack(paths,audit):
    audit=Path(audit).resolve();audit.mkdir(parents=True,exist_ok=False);records=[]
    for p in paths:
        p=Path(p).absolute()
        if not p.resolve().is_relative_to(ROOT/'runs') or p.resolve()!=p:raise ValueError('Outside runs or linked path')
        if any(q.is_symlink() or q.is_junction() for q in [p.parent,*p.parents]):raise ValueError('Linked ancestor')
        if p.suffix not in NATIVE:raise ValueError('Not a native checkpoint')
        z=p.with_name(p.name+'.gz')
        if z.exists():continue
        st=p.stat();h=hashlib.sha256()
        with p.open('rb') as src,z.open('xb') as target:
            with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0,compresslevel=1) as dst:
                while chunk:=src.read(8*1024*1024):h.update(chunk);dst.write(chunk)
        check=hashlib.sha256();n=0
        with gzip.open(z,'rb') as src:
            while chunk:=src.read(8*1024*1024):check.update(chunk);n+=len(chunk)
        assert check.hexdigest()==h.hexdigest() and n==st.st_size
        assert p.stat().st_size==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
        records.append(dict(path=str(p.relative_to(ROOT)),archive=str(z.relative_to(ROOT)),original_sha256=h.hexdigest(),archive_sha256=digest(z),original_bytes=n,archive_bytes=z.stat().st_size))
        (audit/'verified_archives.json').write_text(json.dumps(records,indent=2))
        print(f'Archive verified: {p.relative_to(ROOT)} ({n/1e6:.1f} -> {z.stat().st_size/1e6:.1f} MB)',flush=True)
    (audit/'verified_archives.json').write_text(json.dumps(records,indent=2))
    cleanup=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ROOT/'apply_solver_archives.ps1'),'-Manifest',str(audit/'verified_archives.json')],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    (audit/'cleanup.log').write_bytes(cleanup.stdout)
    cleanup.check_returncode()
    summary=dict(files=len(records),reclaimed_bytes=sum(r['original_bytes']-r['archive_bytes'] for r in records))
    (audit/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def archive_candidate(folder,force=False):
    folder=Path(folder).resolve()
    if not force and (folder.parent.parent/'DEFER_NATIVE_ARCHIVAL').exists():return
    audit=folder/'native_archive'
    if (audit/'summary.json').exists():
        if json.loads((audit/'summary.json').read_text()).get('status')!='deferred_to_end_of_sweep':return
        audit=folder/'native_archive_completed'
        if (audit/'summary.json').exists():return
    summary=json.loads((folder/'summary.json').read_text())
    if summary['status'] in ('checkpoint','evaluation_error'):return
    paths=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix in NATIVE and p.stat().st_size>1_000_000)
    return pack(paths,audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args();out=a.audit.resolve()
    if out.exists():raise ValueError('Use a new audit folder')
    before=inventory()
    # Preserve both recent comparison directories byte-for-byte for their existing audit.
    exclude={'layout_sizing_pilot_v2_20261015','layout_early_restoration_20261015'}
    paths=sorted(p for p in (ROOT/'runs').rglob('*') if p.is_file() and p.suffix in NATIVE and p.stat().st_size>20_000_000 and not (p.with_name(p.name+'.gz')).exists() and not any(v in p.parts for v in exclude))
    protected={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'article_results').rglob('*') if p.is_file()}
    print(json.dumps(dict(before=before,archive_files=len(paths),archive_input_bytes=sum(p.stat().st_size for p in paths))),flush=True)
    result=pack(paths,out)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    (out/'article_sha256.json').write_text(json.dumps(protected,indent=2))
    (out/'before.json').write_text(json.dumps(before,indent=2))
    after=inventory();(out/'after.json').write_text(json.dumps(after,indent=2));print(json.dumps(dict(before=before,after=after,archive=result,article_files_unchanged=len(protected))),flush=True)


if __name__=='__main__':main()
