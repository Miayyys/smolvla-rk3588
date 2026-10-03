
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
from rknn.api import RKNN
out=Path('runs/precision_support');out.mkdir(exist_ok=True,parents=True)
rows=[]
for field,fmt in [('float_dtype','float16'),('float_dtype','bfloat16'),('float_dtype','float32'),('quantized_dtype','w8a8'),('quantized_dtype','w8a16'),('quantized_dtype','w4a16'),('quantized_dtype','w4a4'),('quantized_dtype','w16a16i'),('quantized_dtype','w16a16i_dfp')]:
 r=RKNN(verbose=True,verbose_file=str(out/(fmt+'.log')));row={'field':field,'format':fmt}
 try:
  for stage,fn in [('config',lambda:r.config(target_platform='rk3588',**{field:fmt})),('load',lambda:r.load_onnx(model='runs/tiny_w4a16_probe/tiny.onnx')),('build',lambda:r.build(do_quantization=field=='quantized_dtype',dataset='runs/tiny_w4a16_probe/dataset.txt' if field=='quantized_dtype' else None)),('export',lambda:r.export_rknn(str(out/(fmt+'.rknn'))))]:
   row['stage']=stage;ret=fn()
   if ret!=0:raise RuntimeError(f'{stage}: {ret}')
  row['success']=True
 except BaseException as e:row.update(success=False,error=str(e))
 finally:r.release()
 rows.append(row);(out/'report.json').write_text(json.dumps(rows,indent=2))
