#!/usr/bin/env python3
"""Compare the fixed vision boundary under PyTorch floating-point formats."""

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
import numpy as np
import torch
from qvla.conversion.export_smolvla_vision_onnx import VisionAndConnector
from qvla.evaluation.haq_offline_eval import load_policy
from qvla.haq.offline_actions import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def metrics(actual, reference):
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {"mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.square(delta).mean())),
            "max_abs_error": float(np.abs(delta).max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir', type=Path, default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--boundary-npz', type=Path, default=ROOT/'runs/smolvla_split_contract_v1.npz')
    p.add_argument('--output-dir', type=Path, default=ROOT/'runs/smolvla_vision_float_v1')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    policy, _, _ = load_policy(args)
    policy.model.vlm_with_expert.vlm.set_attn_implementation('eager')
    wrapper = VisionAndConnector(policy).eval()
    with np.load(args.boundary_npz, allow_pickle=False) as z:
        pixel = z['vision_input_0'].copy()
        reference = z['connector_output_0'].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'one pinned vision input, PyTorch format diagnostic',
              'checkpoint_sha256': file_sha256(args.model_dir/'model.safetensors'),
              'boundary_sha256': file_sha256(args.boundary_npz),
              'torch_version': torch.__version__, 'device': args.device, 'results': {}}
    # Reload original FP32 vision weights before each cast to avoid chained rounding.
    originals = {k: v.detach().float().cpu().clone() for k, v in wrapper.state_dict().items()}
    for name, dtype in [('fp32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)]:
        wrapper.to(device=args.device, dtype=torch.float32)
        wrapper.load_state_dict(originals)
        wrapper.to(dtype=dtype)
        x = torch.from_numpy(pixel).to(device=args.device, dtype=dtype)
        with torch.inference_mode():
            out = wrapper(x).float().cpu().numpy()
        path = args.output_dir/f'{name}.npy'
        np.save(path, out, allow_pickle=False)
        report['results'][name] = {'parity_vs_cached_fp': metrics(out, reference),
                                   'output_sha256': file_sha256(path)}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
