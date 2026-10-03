
# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

from rknn.api import RKNN
r=RKNN(verbose=True)
print('CONFIG',r.config(target_platform='rk3588',float_dtype='tfloat32'),flush=True)
print('LOAD',r.load_onnx(model='runs/tiny_w4a16_probe/tiny.onnx'),flush=True)
print('BUILD',r.build(do_quantization=False),flush=True)
print('EXPORT',r.export_rknn('runs/precision_support/tfloat32.rknn'),flush=True)
