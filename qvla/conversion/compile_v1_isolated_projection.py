"""Compile only a V1 exception projection in its requested native RKNN format."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse,hashlib,json,os,time
from pathlib import Path
from rknn.api import RKNN
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--kind',choices=['expert','prefix'],required=True)
p.add_argument('--format',choices=['bfloat16','w16a16i_dfp'],required=True)
a=p.parse_args();root=a.root.resolve();onnx=root/(a.kind+'.onnx')
meta=json.loads((root/(a.kind+'_export.json')).read_text())
h=hashlib.sha256(onnx.read_bytes()).hexdigest()
if h!=meta['onnx_sha256']:raise ValueError('Projection source hash mismatch')
report={'scope':'actual V1 isolated exception, not complete deployment','source_onnx_sha256':h,'requested_format':a.format,'stages':{}}
previous=Path.cwd();os.chdir(root);m=RKNN(verbose=True,verbose_file=str(root/'compile.log'));start=time.monotonic()
try:
 kwargs={'target_platform':'rk3588','optimization_level':3}
 if a.format=='bfloat16':kwargs['float_dtype']='bfloat16'
 else:kwargs.update(quantized_dtype='w16a16i_dfp',quantized_method='layer',float_dtype='float16')
 report['config']=kwargs
 for stage,fn in [('config',lambda:m.config(**kwargs)),('load',lambda:m.load_onnx(model=str(onnx))),('build',lambda:m.build(do_quantization=a.format!='bfloat16',dataset=str(root/(a.kind+'_dataset.txt')) if a.format!='bfloat16' else None)),('export',lambda:m.export_rknn(str(root/'projection.rknn')))]:
  report['stage']=stage;rc=fn();report['stages'][stage]=rc
  if rc:raise RuntimeError(f'{stage}: {rc}')
 report.update(status='compiled',bytes=(root/'projection.rknn').stat().st_size,sha256=hashlib.sha256((root/'projection.rknn').read_bytes()).hexdigest())
except BaseException as e:report.update(status='failed',error=repr(e));raise
finally:
 m.release();os.chdir(previous);report['elapsed_seconds']=time.monotonic()-start
 (root/'compile_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
