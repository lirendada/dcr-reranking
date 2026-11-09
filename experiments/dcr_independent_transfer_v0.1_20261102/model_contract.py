"""Read-only model assets and template checks copied from the frozen factorial runner."""
import hashlib
import importlib.util
import sys
from pathlib import Path
from config import CE_MODEL,STRONG_QWEN,GATE,sha,read

def original_template():
    sys.path.insert(0,str(GATE))
    spec=importlib.util.spec_from_file_location('old_frozen_preparation',GATE/'prepare_external_neural.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.qwen_text

def verify_model_assets() -> dict:
    verified={}
    ce_manifest=read(CE_MODEL/'download_manifest.json')
    ce_files=[]
    for item in ce_manifest['files']:
        path=CE_MODEL/item['name'];actual=sha(path)
        assert actual==item['sha256'],str(path)
        ce_files.append({'name':item['name'],'bytes':path.stat().st_size,'algorithm':'sha256','digest':actual})
    verified['MiniLM']={'directory':str(CE_MODEL),'revision':'233902d25c440f23af6f7d6e94d2946bac0bee0a',
        'download_manifest_sha256':sha(CE_MODEL/'download_manifest.json'),'files':ce_files}
    assets=read(STRONG_QWEN/'model_assets.json');official=read(STRONG_QWEN/'official_model_metadata.json')
    assert assets['revision']=='22e683669bc0f0bd69640a1354a6d0aebcfeede5'
    entries={r['rfilename']:r for r in official['siblings']};files=[]
    for item in assets['files']:
        path=Path(assets['directory'])/item['name'];entry=entries[item['name']]
        size=path.stat().st_size;assert size==entry['size']
        if entry.get('lfs'):
            algorithm='sha256';actual=sha(path);expected=entry['lfs']['sha256']
        else:
            algorithm='git_blob_sha1';h=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            actual=h.hexdigest();expected=entry['blobId']
        assert actual==expected,str(path)
        files.append({'name':item['name'],'bytes':size,'algorithm':algorithm,'digest':actual})
    verified['Qwen']={'directory':assets['directory'],'revision':assets['revision'],
        'official_metadata_sha256':sha(STRONG_QWEN/'official_model_metadata.json'),'files':files}
    return verified

def length_statistics(values: list[int]) -> dict:
    ordered=sorted(values)
    return {'pairs':len(values),'min':ordered[0],'max':ordered[-1],'total':sum(values),
            'p50':ordered[len(ordered)//2],'p95':ordered[int(len(ordered)*.95)]}
