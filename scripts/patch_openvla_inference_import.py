#!/usr/bin/env python3
"""Remove an inference-only import's dependency on the RLDS training stack.

The imported enum is identical: data_utils reexports constants.NormalizationType.
No model forward, preprocessing, normalization, or action code is changed.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-repo',type=Path,required=True)
    args=p.parse_args();repo=args.teacher_repo.resolve()
    source=repo/'experiments/robot/openvla_utils.py'
    old='from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType'
    new='from prismatic.vla.constants import NormalizationType'
    reexport=(repo/'prismatic/vla/datasets/rlds/utils/data_utils.py').read_text()
    if new not in reexport:raise ValueError('Enum origin changed; do not apply this patch')
    text=source.read_text()
    if old in text:
        if text.count(old)!=1:raise ValueError('Unexpected import count')
        source.write_text(text.replace(old,new))
    elif new not in text:raise ValueError('Inference import changed; manual inspection required')
    initializer=repo/'prismatic/vla/__init__.py'
    eager='from .materialize import get_vla_dataset_and_collator'
    lazy='# QVLA inference: load the RLDS training stack only when its API is called.\ndef get_vla_dataset_and_collator(*args, **kwargs):\n    from .materialize import get_vla_dataset_and_collator as implementation\n    return implementation(*args, **kwargs)\n'
    init_text=initializer.read_text()
    if init_text.strip()==eager:initializer.write_text(lazy)
    elif init_text!=lazy:raise ValueError('VLA initializer changed; inspect before lazy import patch')
    record={'revision':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            'changed_file':str(source.relative_to(repo)),'old_import':old,'new_import':new,
            'patched_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'lazy_training_initializer_sha256':hashlib.sha256(initializer.read_bytes()).hexdigest(),
            'scope':'same_enum_import_and_lazy_training_import_no_numeric_algorithm_changes',
            'git_diff':subprocess.check_output(['git','-C',str(repo),'diff','--',str(source.relative_to(repo)),str(initializer.relative_to(repo))],text=True)}
    (repo/'qvla_inference_patch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
