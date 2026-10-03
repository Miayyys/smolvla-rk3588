#!/usr/bin/env python3
"""Run ONE explicit stage of expanded distillation; never chain into QAT."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))
from qvla.paths import source_path

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def commands(args):
    cfg=json.loads(args.config.read_text());student=ROOT/'.venv/bin/python';teacher=ROOT/'.venv-teacher/bin/python'
    script=lambda name:str(source_path(name))
    shared=['--dataset-root','data/libero','--splits','data/libero_splits.json','--partition','config/evaluation_partition_v2.json']
    if args.stage=='prepare':return [str(student),script('prepare_openvla_teacher_inputs.py'),*shared,
        '--samples-per-task',str(cfg['teacher_samples_per_task']),'--output',cfg['teacher_inputs']]
    if args.stage=='label':
        cmd=[str(teacher),script('cache_openvla_teacher.py'),'--teacher-repo','third_party/openvla-oft',
             '--checkpoint','artifacts/teacher/openvla-oft-libero-10','--inputs',cfg['teacher_inputs'],'--output',cfg['teacher_cache']]
        if args.resume:cmd.append('--resume')
        return cmd
    if args.stage=='review':return [str(student),script('review_teacher_labels.py'),'--cache',cfg['teacher_cache'],'--output',cfg['teacher_review']]
    if args.stage=='audit':return [str(student),script('audit_openvla_teacher_labels.py'),*shared,
        '--cache',cfg['teacher_cache'],'--inputs',cfg['teacher_inputs'],'--output',cfg['teacher_cache']+'/contract_audit.json']
    recipe=cfg['fp_distillation'];output='runs/distill_fp_expanded_v3';mode='fp-distill'
    if args.stage=='qat':
        if args.fp_run is None:raise ValueError('Select a verified distilled FP run explicitly before QAT')
        if not args.allow_unverified_backend_diagnostic:
            raise ValueError('Full RKNN mixed-policy parity is pending; only explicit local QAT diagnostics are available')
        mode='qat';output='runs/distill_qat_expanded_v3';recipe=cfg['qat']['proposed_local_recipe']
    cmd=[str(student),script('qat_train_haq.py'),'--mode',mode,'--model-dir','artifacts/model',
         '--vlm-assets-dir','artifacts/smolvlm2_assets',*shared,'--candidate',cfg['student_candidate'],
         '--teacher-cache',cfg['teacher_cache'],'--teacher-review',cfg['teacher_review'],'--require-expanded-teacher',
         '--output-dir',output,'--seed','29']
    for key,value in recipe.items():cmd.extend(['--'+key.replace('_','-'),str(value)])
    if args.stage=='qat':cmd.extend(['--initial-master',str(args.fp_run/'distilled_float_master.safetensors'),'--allow-unverified-backend-diagnostic'])
    if args.resume:cmd.extend(['--resume',output+'/training_state.pt' if args.resume=='latest' else args.resume])
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config/distillation_expanded_v3.json')
    p.add_argument('--stage',choices=('prepare','label','review','audit','fp','qat'),required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',nargs='?',const='latest')
    p.add_argument('--fp-run',type=Path);p.add_argument('--allow-unverified-backend-diagnostic',action='store_true')
    args=p.parse_args()
    if args.resume and args.stage not in ('label','fp','qat'):p.error('Resume is available for labeling and training')
    cmd=commands(args);print(json.dumps({'stage':args.stage,'command':cmd,'training_starts':not args.dry_run and args.stage in ('fp','qat')},indent=2))
    if args.dry_run:return
    if args.stage in ('fp','qat'):
        cfg=json.loads(args.config.read_text());cache=ROOT/cfg['teacher_cache']
        audit=json.loads((cache/'contract_audit.json').read_text())
        from hashlib import sha256
        if not audit['teacher_execution_verified'] or audit['teacher_manifest_sha256']!=sha256((cache/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Audit the exact expanded teacher cache before training')
    subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
