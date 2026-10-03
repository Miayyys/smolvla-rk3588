#!/usr/bin/env python3
"""Build the fixed-shape SmolVLA vision+connector ONNX as FP16 RKNN."""
import argparse
import json
import time
from pathlib import Path

from rknn.api import RKNN


ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx',type=Path,default=ROOT/'runs/smolvla_vision_split_v1/vision_connector.onnx')
    p.add_argument('--output',type=Path,default=ROOT/'runs/smolvla_vision_split_v1/vision_connector_fp16.rknn')
    p.add_argument('--log',type=Path,default=ROOT/'runs/smolvla_vision_split_v1/rknn_build.log')
    p.add_argument('--optimization-level', type=int, choices=(0,1,2,3), default=3)
    p.add_argument('--simulator-input', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--scope', default='fixed-shape vision+connector')
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from hashlib import sha256
    def digest(path):
        h=sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    report={'scope':f'FP16 RKNN compile of {args.scope}; board not yet executed',
            'source_onnx_sha256':digest(args.onnx),'target_platform':'rk3588',
            'float_dtype':'float16','do_quantization':False,
            'optimization_level':args.optimization_level,'status':'started'}
    start=time.perf_counter()
    model=RKNN(verbose=True,verbose_file=str(args.log))
    try:
        for stage,fn in [('config',lambda:model.config(target_platform='rk3588',float_dtype='float16',optimization_level=args.optimization_level)),
                         ('load_onnx',lambda:model.load_onnx(model=str(args.onnx))),
                         ('build',lambda:model.build(do_quantization=False)),
                         ('export',lambda:model.export_rknn(str(args.output)))]:
            report['stage']=stage
            rc=fn()
            if rc!=0:raise RuntimeError(f'{stage} failed with code {rc}')
        report.update(status='passed',output_rknn_sha256=digest(args.output),
                      output_rknn_bytes=args.output.stat().st_size)
        if args.simulator_input:
            import numpy as np
            report['stage'] = 'host_simulator'
            if model.init_runtime() != 0:
                raise RuntimeError('Host simulator init failed')
            sample = np.load(args.simulator_input, allow_pickle=False).astype(np.float32)
            outputs = model.inference(inputs=[sample], data_format=['nchw'])
            if outputs is None or len(outputs) != 1:
                raise RuntimeError('Host simulator inference failed')
            actual = np.asarray(outputs[0], dtype=np.float32)
            np.save(args.output.with_suffix('.simulator.npy'), actual, allow_pickle=False)
            report['simulator_input_sha256'] = digest(args.simulator_input)
            if args.reference:
                reference = np.load(args.reference, allow_pickle=False)
                if actual.shape != reference.shape:
                    raise ValueError('Simulator output shape differs from reference')
                delta = actual.astype(np.float64) - reference.astype(np.float64)
                report['simulator_parity'] = {'mae':float(np.abs(delta).mean()),
                    'rmse':float(np.sqrt(np.square(delta).mean())),
                    'max_abs_error':float(np.abs(delta).max())}
                report['reference_sha256'] = digest(args.reference)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        args.output.with_suffix('.build.json').write_text(json.dumps(report,indent=2)+'\n')
        model.release()
    print(json.dumps(report))


if __name__=='__main__':main()
