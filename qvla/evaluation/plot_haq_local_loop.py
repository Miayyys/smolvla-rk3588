#!/usr/bin/env python3
"""Plot measured local feedback; no illustrative success or board latency data."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text())
    reports=[json.loads(f.read_text()) for f in sorted(args.run.glob('round*/report.json'))]
    x=list(range(1,len(reports)+1))
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    panels=[([100*r['compression_fraction'] for r in reports], 'Actual checkpoint compression (%)'),
            ([r['metrics']['task_macro']['chunk_mae_vs_fp'] for r in reports], 'Full action chunk MAE vs FP'),
            ([r['cost']['total_ms'] for r in reports], 'Sum of isolated RK3588 costs (ms, proxy)'),
            ([r['reward'] for r in reports], 'Offline quality + lookup reward')]
    for ax,(values,label) in zip(axes.flat,panels):
        ax.plot(x,values,'o-',linewidth=1)
        ax.set(xlabel='Sequential candidate evaluation',ylabel=label)
        ax.grid(alpha=.25)
    axes[0,0].axhline(40,color='red',linestyle='--',label='Hard 40% threshold')
    axes[0,0].legend()
    fig.suptitle(f"Local real action feedback: {s['rounds']} RL updates, {len(reports)} candidates\n"
                 '40 development observations each; no closed-loop success/whole-board latency measured')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(args.output.with_suffix('.'+ext),dpi=160)


if __name__=='__main__':main()
