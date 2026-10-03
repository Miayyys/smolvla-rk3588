"""Compile real selected language weights in FP16 before quantization tuning."""
import argparse
import json
import time
from pathlib import Path
from rkllm.api import RKLLM

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
report = {'scope': 'real language conversion; original attention mask not yet supported',
          'stages': {}, 'mask_equivalence': False}
start = time.monotonic()
try:
    model = RKLLM()
    for name, fn in [
        ('load', lambda: model.load_huggingface(str(a.root / 'hf_model'), dtype='float32')),
        ('build', lambda: model.build(do_quantization=False, target_platform='rk3588',
                                    num_npu_core=3, max_context=256)),
        ('export', lambda: model.export_rkllm(str(a.root / 'language_fp16.rkllm'),
                                             export_embedding=False))]:
        ret = fn()
        report['stages'][name] = ret
        if ret != 0:
            raise RuntimeError(f'{name} returned {ret}')
    report['status'] = 'compiled_not_equivalent_deployment'
except BaseException as e:
    report.update(status='failed', error=repr(e))
    raise
finally:
    report['elapsed_seconds'] = time.monotonic() - start
    (a.root / 'compile_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
