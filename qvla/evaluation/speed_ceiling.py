"""Isolated speed screen. Reduced steps/alternate graphs have no quality guarantee."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def unchecked_partitions(self, inputs):
    values = dict(zip(self.input_names, inputs))
    self.last_timings = []
    for model, part in self.models:
        arrays = []
        for name in part['input_names']:
            x = values[name]
            if name in self.static_inputs and x is self.static_inputs[name]:
                arrays.append(self.static_arrays[(name, part['input_dtypes'][name])])
            else:
                arrays.append(self.prepare_input(x, part['input_dtypes'][name], name))
        result = model.inference(inputs=arrays, data_format=['nchw'] * len(arrays))
        if result is None or len(result) != len(part['output_names']):
            raise RuntimeError('SDK execution failed')
        values.update(zip(part['output_names'], result))
    return [values[name] for name in self.output_names]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--options', default='{}')
    p.add_argument('--steps', default='10,5,3,1')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--fast-checks', action='store_true')
    p.add_argument('--expert-layers', type=int, help='Input adapter for a structurally shortened monolithic expert')
    p.add_argument('--image-size', type=int, help='Experimental preprocessor size; must match compiled graph')
    p.add_argument('--native-single-camera', action='store_true', help='Omit second view entirely; requires matching single-view language/expert graphs')
    p.add_argument('--single-camera', action='store_true', help='Run vision once, duplicate features; keep fixed two-view prefix graph')
    a = p.parse_args()
    from qvla.runtime.smolvla_board_runtime import BoardSmolVLA
    from qvla.runtime.smolvla_numpy_glue import time_embedding
    if a.fast_checks:
        from qvla.runtime.smolvla_rknn_partitions import RKNNPartitions
        RKNNPartitions.infer = unchecked_partitions
    with np.load(a.inputs, allow_pickle=False) as z:
        raw = {k: z[k][0].copy() for k in z.files}
    model = BoardSmolVLA(a.root, options=json.loads(a.options))
    if a.image_size is not None:
        if a.image_size < 64 or a.image_size % 64:
            raise ValueError('Probe image size must be a positive multiple of 64')
        model.pre.width = a.image_size
        model.pre.height = a.image_size
    # Production runtime pins monolithic experts to core 0. Honor the probe
    # option explicitly here without changing the validated production path.
    expert_core = model.options.get('expert_core', '0')
    if 'expert' not in model.partition_models and expert_core != '0':
        from rknnlite.api import RKNNLite
        dm = json.loads((a.root / 'deployment_manifest.json').read_text())
        model.models[2].release()
        m = RKNNLite()
        model.models[2] = m
        if m.load_rknn(str(a.root / dm['graphs'][2]['filename'])) or m.init_runtime(core_mask=model.core_mask(expert_core)):
            model.close()
            raise RuntimeError('Probe expert initialization failed')
    if a.expert_layers is not None:
        if not 1 <= a.expert_layers <= 16 or 'expert' in model.partition_models:
            raise ValueError('Depth probe requires a monolithic expert with 1..16 layers')
        class SelectedKV:
            def __init__(self, inner, layers):
                self.inner = inner
                self.inputs = 3 + 2 * layers
            def inference(self, inputs, **kwargs):
                kwargs['data_format'] = kwargs['data_format'][:self.inputs]
                return self.inner.inference(inputs=inputs[:self.inputs], **kwargs)
            def release(self):
                self.inner.release()
        model.models[2] = SelectedKV(model.models[2], a.expert_layers)
    if a.native_single_camera:
        if a.single_camera or model.vision_executor:
            raise ValueError('Native single view requires one context and no duplication')
        class SinglePreprocessor:
            def __init__(self, inner):
                self.inner = inner
            def __call__(self, *args, **kwargs):
                x = self.inner(*args, **kwargs)
                x['images'] = x['images'][:1]
                x['image_masks'] = x['image_masks'][:1]
                return x
        model.pre = SinglePreprocessor(model.pre)
    if a.single_camera:
        if model.vision_executor:
            raise ValueError('Single-camera probe requires one vision context')
        class ReuseSecondView:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.calls = 0
                self.cached = None
            def inference(self, **kwargs):
                if self.calls % 2 == 0:
                    self.cached = self.wrapped.inference(**kwargs)
                self.calls += 1
                return self.cached
            def release(self):
                self.wrapped.release()
        model.models[0] = ReuseSecondView(model.models[0])
    report = {'native_single_camera': a.native_single_camera, 'structural_expert_layers': a.expert_layers, 'experimental_image_size': a.image_size, 'single_camera_duplicate_feature_probe': a.single_camera, 'options': model.options, 'hashes': model.hashes,
              'quality_validated': False, 'partition_finite_checks_disabled': a.fast_checks,
              'input_sha256': hashlib.sha256(a.inputs.read_bytes()).hexdigest(), 'cases': []}
    try:
        for steps in map(int, a.steps.split(',')):
            if steps < 1: raise ValueError('steps must be positive')
            model.config['num_steps'] = steps
            model.time_embeddings = tuple(time_embedding(np.array([1. + i * (-1. / steps)], np.float32),
                model.config['expert_hidden_size'], model.config['min_period'], model.config['max_period']) for i in range(steps))
            model.predict(raw)
            samples = []
            for i in range(a.repeats):
                begin = time.perf_counter()
                y, t = model.predict(raw)
                wall_ms = (time.perf_counter() - begin) * 1000
                if a.single_camera or a.native_single_camera:
                    t['calls']['vision'] = 1
                samples.append({**t, 'wall_ms': wall_ms, 'finite': bool(np.isfinite(y).all()),
                                'shape': list(y.shape), 'output_sha256': hashlib.sha256(y.tobytes()).hexdigest()})
            values = [s['wall_ms'] for s in samples]
            case = {'steps': steps, 'p50_ms': float(np.median(values)), 'min_ms': min(values), 'samples': samples}
            report['cases'].append(case)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'steps': steps, 'p50_ms': case['p50_ms'], 'min_ms': min(values)}), flush=True)
    finally:
        model.close()

if __name__ == '__main__':
    main()
