"""Convert the tiny fixture with recorded SDK return codes."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import json
from pathlib import Path
from rkllm.api import RKLLM

root=Path(__file__).resolve().parents[2]/'runs/rkllm_dump_probe_v1'
m=RKLLM()
report={'scope':'tiny dump probe conversion, not selected QAT deployment','stages':{}}
try:
    for name,fn in [('load',lambda:m.load_huggingface(str(root/'hf_model'),device='cpu',dtype='float32')),
                    ('build',lambda:m.build(do_quantization=True,quantized_dtype='w8a8',
                        target_platform='rk3588',num_npu_core=1,max_context=128,
                        dataset=str(root/'dataset.json'),optimization_level=0)),
                    ('export',lambda:m.export_rkllm(str(root/'tiny.rkllm')))]:
        rc=fn();report['stages'][name]=rc
        if rc!=0:raise RuntimeError(f'{name}: {rc}')
    report['status']='compiled'
except BaseException as e:
    report.update(status='failed',error=repr(e));raise
finally:
    (root/'compile_report.json').write_text(json.dumps(report,indent=2)+'\n')
