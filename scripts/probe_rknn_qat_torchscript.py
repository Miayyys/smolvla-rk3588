#!/usr/bin/env python3
"""Convert one actual SmolVLA Linear via PyTorch eager QAT to int8 TorchScript.

No training is performed. This only tests the RKNN conversion interface.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import (DeQuantStub, QuantStub, convert,
                                    get_default_qat_qconfig, prepare_qat)

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.quant = QuantStub()
        self.linear = copy.deepcopy(linear).float()
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.linear(self.quant(x)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--module", default=("model.vlm_with_expert.vlm.model."
                        "vision_model.encoder.layers.10.mlp.fc1"))
    parser.add_argument("--calibration-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.quantized.engine = "fbgemm"
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cpu"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    source = policy.get_submodule(args.module)
    if not isinstance(source, nn.Linear):
        raise TypeError("Expected Linear")
    paths = [Path(line) for line in args.calibration_inputs.read_text().splitlines()
             if line.strip()]
    if not paths:
        raise ValueError("No calibration inputs")
    model = QuantizedLinear(source).train()
    model.qconfig = get_default_qat_qconfig("fbgemm")
    qconfig_repr = str(model.qconfig)
    prepare_qat(model, inplace=True)
    sample = None
    with torch.no_grad():
        for path in paths:
            arr = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
            model(arr)
            if sample is None:
                sample = arr
    model.eval()
    with torch.no_grad():
        fake_output = model(sample).numpy()
    convert(model, inplace=True)
    if not isinstance(model.linear, torch.ao.nn.quantized.Linear):
        raise ValueError(f"Linear was not quantized: {type(model.linear)}")
    with torch.no_grad():
        converted = model(sample).numpy()
        traced = torch.jit.trace(model, sample)
        traced.save(str(args.output_dir / "qat_untrained_int8.pt"))
    np.save(args.output_dir / "sample_input.npy", sample.numpy(), allow_pickle=False)
    np.save(args.output_dir / "converted_output.npy", converted, allow_pickle=False)
    report = {"scope": "untrained_single_real_checkpoint_linear_qat_conversion_probe",
              "module": args.module, "calibration_count": len(paths),
              "input_shape": list(sample.shape),
              "qconfig": qconfig_repr,
              "converted_linear_type": str(type(model.linear)),
              "packed_weight_dtype": str(model.linear.weight().dtype),
              "fake_to_converted_mae": float(np.mean(np.abs(fake_output - converted))),
              "fake_to_converted_max_abs": float(np.max(np.abs(fake_output - converted))),
              "torchscript_bytes": (args.output_dir / "qat_untrained_int8.pt").stat().st_size,
              "trained": False, "rknn_conversion": "not_measured"}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
