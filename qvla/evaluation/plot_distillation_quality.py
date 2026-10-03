#!/usr/bin/env python3
"""Plot measured valid action chunks; offline errors are not task success."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path('runs/distill_real_teacher_evidence_v1'))
    p.add_argument('--output',type=Path,default=Path('docs/images/distill_qat_quality_v1'))
    p.add_argument('--closed-loop',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    inputs=[args.evidence/'development_lr1e5/full_chunks.json',args.evidence/'development_lr1e7/full_chunks.json']
    hi,lo=[json.loads(f.read_text())['metrics'] for f in inputs]
    rows=[('Original FP',hi['original_fp']),('Original v2',hi['original_v2']),
          ('FP lr=1e-5',hi['distilled_fp']),('QAT lr=1e-5',hi['distilled_v2_qat']),
          ('FP lr=1e-7',lo['distilled_fp']),('QAT lr=1e-7',lo['distilled_v2_qat'])]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for label,r in rows:
        axes[0].plot(range(50),r['per_timestep_mae_vs_demonstration'],label=label)
    axes[0].set(xlabel='Predicted action timestep',ylabel='MAE vs recorded demonstration',title='40 development observations; valid timesteps only')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    bars=axes[1].barh([x[0] for x in rows],[x[1]['valid_chunk_mae_vs_demonstration'] for x in rows])
    axes[1].bar_label(bars,fmt='%.5f',padding=3,fontsize=8)
    axes[1].set(xlabel='Task macro MAE over valid action chunks',title='Offline diagnostic; not task success')
    axes[1].invert_yaxis();axes[1].set_xlim(0,.10)
    for ext in ['png','svg']:fig.savefig(args.output/f'full_chunks.{ext}',dpi=180)
    plt.close(fig)
    if args.closed_loop:
        measured=json.loads(args.closed_loop.read_text())
        if measured['total_episodes']!=30 or not measured['paired_initial_state_images_and_noise_verified']:
            raise ValueError('Expected the audited paired 30-episode panel')
        names=['original_fp','original_v2','fp_lr1e5','qat_lr1e5','fp_lr1e7','qat_lr1e7']
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        bars=ax.barh([r[0] for r in rows],[measured['successes'][n] for n in names])
        ax.bar_label(bars,fmt='%d / 5',padding=3)
        ax.set(xlim=(0,5.6),xticks=range(6),xlabel='Successful task episodes',
               title='Five predefined cases; one paired initial state\nSmall diagnostic, not overall success rate')
        ax.invert_yaxis()
        for ext in ['png','svg']:fig.savefig(args.output/f'closed_loop.{ext}',dpi=180)
        plt.close(fig)
        inputs.append(args.closed_loop)
    provenance={'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
                'scope':'measured_offline_action_errors_not_closed_loop_quality'}
    (args.output/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()
