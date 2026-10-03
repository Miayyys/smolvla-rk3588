#!/usr/bin/env python3
"""Plot only measured vision deployment diagnostic data."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/qvla-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    base=ROOT/'runs/smolvla_vision_split_v1'
    sources={}
    def read(path):
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    original=read(base/'board_report_default_232.json')
    board=read(base/'board_report_nhwc.json')
    ort=read(base/'onnx_parity.json')
    local=read(ROOT/'runs/smolvla_vision_float_v1/report.json')
    simulator=read(base/'vision_opt3_sim_fp16.build.json')
    stages=read(base/'board_stages.json')
    panel=read(ROOT/'runs/smolvla_board_vision_panel_v1/action_report.json')
    names=['ONNX FP32','Torch FP16','Torch BF16','RKNN simulator','RK3588 fixed layout','RK3588 old layout']
    maes=[ort['mae'],local['results']['fp16']['parity_vs_cached_fp']['mae'],
          local['results']['bf16']['parity_vs_cached_fp']['mae'],simulator['simulator_parity']['mae'],
          board['parity_vs_reference']['mae'],original['parity_vs_reference']['mae']]
    order=[1,2,3,4,5,6,0]
    stage_names=['Patch+position','Layer 0','Layer 3','Layer 7','Layer 11','Post LayerNorm','Connector']
    errors=[stages['outputs'][i]['relative_rmse']*100 for i in order]
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    axs[0].barh(names,maes,color=['#5899da']*4+['#4caa79','#d87965'])
    axs[0].set_xscale('log');axs[0].set_xlabel('Feature MAE vs original FP (log scale)')
    axs[0].set_title('Same camera input; one observation')
    axs[1].plot(range(7),errors,'o-',color='#4477aa')
    axs[1].set_xticks(range(7),stage_names,rotation=40,ha='right')
    axs[1].set_ylabel('RMSE / reference RMS (%)')
    axs[1].set_title('Instrumented RK3588 vision graph')
    axs[2].bar(range(40),panel['per_observation_action_mae'],color='#4caa79')
    axs[2].set_xlabel('Development observation / task index')
    axs[2].set_ylabel('Final action MAE vs original FP')
    axs[2].set_title('Real board vision + GPU prefix/expert')
    axs[2].text(0.98,0.95,f"{len(panel['gripper_sign_disagreements'])} gripper sign changes / 2000 actions",
                ha='right',va='top',transform=axs[2].transAxes,fontsize=9)
    fig.suptitle('Pinned checkpoint; fixed action seeds; offline diagnostics, no closed-loop success claim',fontsize=11)
    fig.tight_layout()
    out=ROOT/'docs/images/vision_deployment_diagnostic_v1'
    fig.savefig(out.with_suffix('.png'),dpi=180)
    fig.savefig(out.with_suffix('.svg'))
    record={'scope':'measured diagnostics only','source_report_sha256':sources,
            'feature_mae':dict(zip(names,maes)),'stage_relative_rmse_pct':dict(zip(stage_names,errors)),
            'development_action_mae':panel['per_observation_action_mae']}
    out.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
