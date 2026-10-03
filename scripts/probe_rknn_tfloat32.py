from rknn.api import RKNN
r=RKNN(verbose=True)
print('CONFIG',r.config(target_platform='rk3588',float_dtype='tfloat32'),flush=True)
print('LOAD',r.load_onnx(model='runs/tiny_w4a16_probe/tiny.onnx'),flush=True)
print('BUILD',r.build(do_quantization=False),flush=True)
print('EXPORT',r.export_rknn('runs/precision_support/tfloat32.rknn'),flush=True)
