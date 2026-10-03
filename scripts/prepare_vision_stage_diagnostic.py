#!/usr/bin/env python3
"""Expose coarse vision stages in the pinned ONNX and save CPU references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/smolvla_vision_split_v1'))
    args = p.parse_args()
    source = args.root/'vision_connector.onnx'
    m = onnx.load(source)
    names = ['features', '/Add_output_0'] + [
        f'/encoder/layers.{i}/Add_1_output_0' for i in (0,3,7,11)] + [
        '/post_layernorm/LayerNormalization_output_0']
    available = {o for n in m.graph.node for o in n.output}
    if not set(names) <= available:
        raise ValueError('Expected pinned graph stage is missing')
    del m.graph.output[:]
    for name in names:
        shape = [1,64,960] if name == 'features' else [1,1024,768]
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    onnx.checker.check_model(m)
    path = args.root/'vision_stages.onnx'
    onnx.save(m, path)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    pixel = np.load(args.root/'vision_input_0.npy', allow_pickle=False)
    values = sess.run(names, {sess.get_inputs()[0].name: pixel})
    np.savez_compressed(args.root/'vision_stages_reference.npz',
                        **{f'output_{i}': v for i,v in enumerate(values)})
    report = {'scope':'instrumented ONNX vision stages; graph may change RKNN fusion',
              'source_onnx_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'outputs':[{'index':i,'name':name,'shape':list(v.shape)}
                         for i,(name,v) in enumerate(zip(names,values))]}
    (args.root/'vision_stages.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
