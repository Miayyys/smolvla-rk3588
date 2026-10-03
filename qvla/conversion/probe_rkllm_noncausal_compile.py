"""Diagnostic writer metadata request; does NOT prove noncausal execution."""

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
from rkllm.base.converter import LLMWriter

root=Path(__file__).resolve().parents[2]/'runs/rkllm_dump_probe_v1'
original=LLMWriter.write_header_to_file
calls=[]
def header(self,*args,**kwargs):
    self.add_causal_attention(False)
    calls.append(True)
    print('DIAGNOSTIC: writer causal_attention=False',flush=True)
    return original(self,*args,**kwargs)

report={'scope':'noncausal metadata probe, real attention semantics must be measured'}
LLMWriter.write_header_to_file=header
try:
    m=RKLLM()
    assert m.load_huggingface(str(root/'hf_model'),dtype='float32')==0
    assert m.build(do_quantization=False,target_platform='rk3588',num_npu_core=1,max_context=128)==0
    assert m.export_rkllm(str(root/'tiny_noncausal_fp16.rkllm'))==0
    assert calls
    report.update(status='compiled',writer_hook_calls=len(calls))
except BaseException as e:
    report.update(status='failed',error=repr(e));raise
finally:
    LLMWriter.write_header_to_file=original
    (root/'noncausal_compile_report.json').write_text(json.dumps(report,indent=2)+'\n')
