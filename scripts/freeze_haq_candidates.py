#!/usr/bin/env python3
"""Freeze local diagnostic candidates without treating them as deployed models."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={'v1':'haq_rl_trial_200x4_rl','v2':'haq_exploration_min200_v1_rl'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    for version,source in SOURCES.items():
        run=ROOT/'runs'/source
        summary=json.loads((run/'summary.json').read_text());best=summary['best_by_proxy']
        selected=run/f"round{best['round']:03d}_candidate{best['candidate']:02d}"
        if sha(selected/'model.safetensors')!=best['checkpoint_sha256']:
            raise ValueError('Selected candidate hash changed')
        target=ROOT/'artifacts/candidates'/version
        target.mkdir(parents=True,exist_ok=True)
        files={'model.safetensors':selected/'model.safetensors',
               'assignment.json':selected/'assignment.json','candidate_report.json':selected/'report.json',
               'space.json':run/'space.json','calibration.json':run/'calibration.json',
               'tables_snapshot.json':run/'tables_snapshot.json'}
        hashes={}
        for name,path in files.items():
            digest=sha(path)
            if (target/name).exists():
                if sha(target/name)!=digest:raise ValueError('Refuse to replace frozen candidate')
            else:shutil.copy2(path,target/name)
            hashes[name]=digest
        manifest={'version':version,'source_run':source,'source_checkpoint_sha256':summary['identity']['checkpoint_sha256'],
                  'selected_round':best['round'],'selected_candidate':best['candidate'],
                  'actual_model_bytes':best['actual_model_bytes'],'compression_fraction':best['compression_fraction'],
                  'files_sha256':hashes,'deployment_verified':False,
                  'scope':'frozen_local_mixed_precision_diagnostic_not_RKNN_model'}
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        public={**manifest,'artifact_directory':f'artifacts/candidates/{version}',
                'assignment':json.loads((target/'assignment.json').read_text())}
        (ROOT/'config'/f'haq_candidate_{version}.json').write_text(json.dumps(public,indent=2)+'\n')
        print(version,best['actual_model_bytes'],best['checkpoint_sha256'],flush=True)


if __name__=='__main__':main()
