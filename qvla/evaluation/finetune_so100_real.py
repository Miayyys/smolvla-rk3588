"""Real-demonstration SmolVLA fine-tuning with episode-isolated validation.

Uses LeRobot's policy, processors, flow-matching loss and padding contract.
The packed JPEG cache avoids repeated AV1 seeking; alignment is checked against
LeRobot's own decoder before any training. No test frames enter training/stats.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def emit(out, event):
    event['time_unix'] = time.time()
    with (out / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    print(json.dumps(event), flush=True)


def prepare(cfg, out):
    import av
    import pyarrow.parquet as pq
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(cfg['dataset_root'])
    partition = json.loads(Path(cfg['partition']).read_text())
    table_path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(table_path)
    eps = np.asarray(table['episode_index'])
    splits = partition['splits']
    assert all(not set(splits[a]) & set(splits[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    train_rows = np.flatnonzero(np.isin(eps, splits['train']))
    train = np.isin(eps, splits['train'])
    states = np.asarray(table['observation.state'].to_pylist(), dtype='float32')
    actions = np.asarray(table['action'].to_pylist(), dtype='float32')
    stats = {key: {'mean': a[train].mean(0).tolist(), 'std': np.maximum(a[train].std(0), 1e-6).tolist()}
             for key, a in [('observation.state', states), ('action', actions)]}
    end_rows = np.empty(len(eps), dtype='int64')
    validation_rows = []
    for ep in sorted(set(eps.tolist())):
        rows = np.flatnonzero(eps == ep)
        assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
        end_rows[rows] = rows[-1]
        if ep in splits['validation']:
            validation_rows.extend(np.linspace(rows[0], max(rows[0], rows[-1] - 49), cfg['validation_frames_per_episode'], dtype=int).tolist())
    metadata = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    timestamps = np.asarray(table['timestamp'], dtype='float64')
    expected = timestamps.copy()
    for episode in metadata:
        ep = episode['episode_index']
        expected[eps == ep] += episode['videos/observation.images.top/from_timestamp']
    video = root / 'videos/observation.images.top/chunk-000/file-000.mp4'
    cache_path = out / 'top_frames.jpeg.bin'
    offsets, lengths = [], []
    max_alignment_error = 0.0
    samples = {}
    selected_checks = [0, len(eps) // 2, len(eps) - 1]
    with av.open(str(video)) as container, (out / 'top_frames.partial').open('wb') as target:
        container.streams.video[0].thread_type = 'AUTO'
        count = 0
        for frame in container.decode(video=0):
            if count >= len(eps):
                raise ValueError('Video has more frames than table; need explicit timestamp mapping')
            error = abs(float(frame.pts * frame.time_base) - expected[count])
            max_alignment_error = max(max_alignment_error, error)
            if error > 1e-4:
                raise ValueError(f'Video/table timestamp mismatch at {count}: {error}')
            array = frame.to_ndarray(format='rgb24')
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format='JPEG', quality=95, subsampling=0)
            data = buffer.getvalue()
            offsets.append(target.tell()); lengths.append(len(data)); target.write(data)
            if count in selected_checks:
                samples[count] = array
            count += 1
            if count % 2000 == 0:
                emit(out, {'stage': 'cache', 'frames': count, 'total': len(eps)})
    if count != len(eps):
        raise ValueError('Video/table frame count mismatch')
    (out / 'top_frames.partial').replace(cache_path)
    # The official reader independently validates RGB decoding/time offsets.
    dataset = LeRobotDataset('lerobot/svla_so100_pickplace', root=root.resolve(), video_backend='pyav', return_uint8=True)
    with cache_path.open('rb') as stream:
        rgb_errors = []
        for row in selected_checks:
            official = dataset[row]['observation.images.top'].permute(1, 2, 0).numpy()
            if not np.array_equal(official, samples[row]):
                raise ValueError('Sequential decoder differs from official LeRobot RGB frame')
            stream.seek(offsets[row]); image = np.asarray(Image.open(io.BytesIO(stream.read(lengths[row]))).convert('RGB'))
            rgb_errors.append(float(np.abs(image.astype('float32') - official).mean()))
    np.savez(out / 'data.npz', states=states, actions=actions, episodes=eps, end_rows=end_rows,
             train_rows=train_rows, validation_rows=validation_rows, offsets=offsets, lengths=lengths)
    (out / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    manifest = {'config': cfg, 'splits': splits, 'frames': len(eps), 'train_frames': len(train_rows),
                'validation_rows': validation_rows, 'dataset_sha256': sha(table_path),
                'video_sha256': sha(video), 'partition_sha256': sha(cfg['partition']),
                'cache_sha256': sha(cache_path), 'cache_bytes': cache_path.stat().st_size,
                'max_timestamp_error_s': max_alignment_error, 'jpeg_rgb_mae_0_255': rgb_errors,
                'official_decoder_exact_checks': selected_checks,
                'stats_sha256': sha(out / 'stats.json')}
    (out / 'prepared.json').write_text(json.dumps(manifest, indent=2) + '\n')
    emit(out, {'stage': 'prepared', 'train_frames': len(train_rows), 'cache_bytes': manifest['cache_bytes']})


class Frames:
    def __init__(self, out, task):
        self.data = np.load(out / 'data.npz')
        self.stream = (out / 'top_frames.jpeg.bin').open('rb')
        self.task = task

    def batch(self, rows):
        import torch
        from PIL import Image
        images, actions, masks = [], [], []
        for row in rows:
            self.stream.seek(int(self.data['offsets'][row]))
            image = Image.open(io.BytesIO(self.stream.read(int(self.data['lengths'][row])))).convert('RGB')
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
            indices = row + np.arange(50)
            masks.append(indices > self.data['end_rows'][row])
            actions.append(self.data['actions'][np.minimum(indices, self.data['end_rows'][row])])
        return {'observation.images.camera1': torch.stack(images).float() / 255,
                'observation.state': torch.from_numpy(self.data['states'][rows].copy()),
                'action': torch.from_numpy(np.stack(actions)),
                'action_is_pad': torch.from_numpy(np.stack(masks)), 'task': [self.task] * len(rows)}


def train(cfg, out, smoke, resume=False):
    import torch
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation.haq_offline_eval import load_policy
    from safetensors.torch import save_file

    prepared = json.loads((out / 'prepared.json').read_text())
    if prepared['config'] != cfg or prepared['partition_sha256'] != sha(cfg['partition']):
        raise ValueError('Prepared data/config identity differs')
    if (out / 'progress.json').exists() and not smoke and not resume:
        raise ValueError('Training already exists; preserve it instead of overwriting')
    stats = json.loads((out / 'stats.json').read_text())
    if sha(out / 'stats.json') != prepared['stats_sha256']:
        raise ValueError('Normalization statistics changed')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    policy, _, _ = load_policy(SimpleNamespace(model_dir=Path(cfg['model_dir']).resolve(),
        vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float()
    policy.config.resize_imgs_with_padding = (cfg['image_size'], cfg['image_size'])
    # Keep only the configured cameras in the model interface; missing views are not duplicated.
    cameras = list(cfg.get('camera_map', {'observation.images.camera1': None}))
    policy.config.input_features = {k: v for k, v in policy.config.input_features.items()
                                    if k in ['observation.state', *cameras]}
    policy.config.empty_cameras = 0
    pre, post = make_pre_post_processors(policy.config, cfg['model_dir'],
        preprocessor_overrides={
            'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
            'device_processor': {'device': 'cuda'}, 'normalizer_processor': {'stats': stats}},
        postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    if cfg.get('evaluation') == 'scripted_sim':
        from qvla.evaluation.finetune_so101_scripted import ScriptedFrames
        frames = ScriptedFrames(cfg, out)
    else:
        frames = Frames(out, cfg['task'])
    parameters = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    if any('.vlm.' in n for n, _ in parameters):
        raise ValueError('VLM unexpectedly trainable')
    trained_layers = sorted({int(n.split('.layers.')[1].split('.')[0]) for n, _ in parameters if 'lm_expert.layers.' in n})
    assert trained_layers == list(range(16))
    (out / 'trainable.json').write_text(json.dumps({'names': [n for n, _ in parameters],
        'parameters': sum(p.numel() for _, p in parameters), 'expert_layers': trained_layers,
        'model_source_sha256': sha(Path(cfg['model_dir']) / 'model.safetensors')}, indent=2) + '\n')
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=cfg['learning_rate'], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg['weight_decay'])

    def evaluate(step):
        old_rng = torch.get_rng_state(); old_cuda_rng = torch.cuda.get_rng_state()
        joint_errors = []
        policy.eval()
        try:
            for row in frames.data['validation_rows']:
                raw = frames.batch([int(row)])
                target = raw['action'].numpy()[0].copy()
                mask = ~raw['action_is_pad'].numpy()[0]
                torch.manual_seed(100000 + int(row))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    batch = pre(raw)
                    prediction = post(policy.predict_action_chunk(batch)).float().cpu().numpy()[0]
                joint_errors.append(np.abs(prediction[mask] - target[mask]))
            mae = np.concatenate(joint_errors).mean(0)
            normalized = float(np.mean(mae / np.asarray(stats['action']['std'])))
            result = {'stage': 'validation', 'step': step, 'normalized_action_mae': normalized,
                      'per_joint_mae': mae.tolist(), 'units': 'degrees for 5 arm joints; gripper percentage points',
                      'observations': len(joint_errors), 'scope': 'offline held-out demonstrations; not task success'}
            emit(out, result)
            return normalized
        finally:
            torch.set_rng_state(old_rng); torch.cuda.set_rng_state(old_cuda_rng); policy.train()

    def save_model(name):
        folder = out / name; folder.mkdir(exist_ok=True)
        temporary = folder / 'model.partial.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}, str(temporary))
        temporary.replace(folder / 'model.safetensors')
        policy.config.save_pretrained(folder)
        pre.save_pretrained(folder); post.save_pretrained(folder)

    started = time.perf_counter()
    best_step = stale = start_step = 0
    if resume:
        from safetensors.torch import load_file
        checkpoint = torch.load(out / 'training_state.pt', map_location='cuda', weights_only=False)
        if checkpoint['config'] != cfg or checkpoint['last_weights_sha256'] != sha(out / 'last/model.safetensors'):
            raise ValueError('Resume identity or saved weights changed')
        policy.load_state_dict(load_file(str(out / 'last/model.safetensors'), device='cuda'), strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state = checkpoint['rng']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
        best, best_step = checkpoint['best_metric'], checkpoint['best_step']
        stale, start_step = checkpoint['stale_evaluations'], checkpoint['step']
        emit(out, {'stage': 'resumed', 'step': start_step})
    else:
        best = float('inf') if smoke else evaluate(0)
    first_param = next(p for n, p in parameters if 'action_out_proj.weight' in n)
    initial = first_param.detach().clone()
    for step in range(start_step + 1, (2 if smoke else cfg['steps']) + 1):
        policy.train(); optimizer.zero_grad(set_to_none=True)
        progress = min(1.0, max(0.0, (step - cfg['warmup_steps']) / (cfg['steps'] - cfg['warmup_steps'])))
        lr = (cfg['minimum_learning_rate'] + 0.5 * (cfg['learning_rate'] - cfg['minimum_learning_rate']) * (1 + math.cos(math.pi * progress)))
        if step <= cfg['warmup_steps']:
            lr = cfg['learning_rate'] * step / cfg['warmup_steps']
        for group in optimizer.param_groups: group['lr'] = lr
        losses = []
        for _ in range(1 if smoke else cfg['gradient_accumulation']):
            rows = rng.choice(frames.data['train_rows'], cfg['batch_size'], replace=True)
            batch = pre(frames.batch(rows))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = policy(batch)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            (loss / (1 if smoke else cfg['gradient_accumulation'])).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], cfg['gradient_clip'])
        if not torch.isfinite(norm) or float(norm) == 0: raise RuntimeError('Invalid gradient norm')
        if step == 1:
            stage_grads = {str(i): sum(float(p.grad.abs().sum()) for n, p in parameters
                if f'lm_expert.layers.{i}.' in n and p.grad is not None) for i in range(16)}
            if any(v <= 0 or not math.isfinite(v) for v in stage_grads.values()):
                raise RuntimeError('Missing expert layer gradient')
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen weights have gradients')
            emit(out, {'stage': 'gradient_audit', 'all_16_expert_layers_positive': True,
                       'per_layer_gradient_l1': stage_grads, 'frozen_gradients_absent': True})
        optimizer.step()
        record = {'stage': 'smoke' if smoke else 'training', 'step': step, 'steps': cfg['steps'],
                  'loss': float(np.mean(losses)), 'learning_rate': lr, 'gradient_norm': float(norm),
                  'elapsed_seconds': time.perf_counter() - started,
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
        if smoke or step % 10 == 0 or step == 1:
            emit(out, record)
        if not smoke:
            (out / 'progress.json').write_text(json.dumps(record, indent=2) + '\n')
        if smoke:
            continue
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            metric = evaluate(step)
            if metric < best:
                best, best_step, stale = metric, step, 0
                save_model('best')
            else:
                stale += 1
            if not cfg.get('save_last', True):
                # Disk-limited runs keep only the selected checkpoint; no resume state.
                if step >= cfg['minimum_steps'] and stale >= cfg['patience_evaluations']:
                    emit(out, {'stage': 'early_stop', 'step': step, 'best_step': best_step})
                    break
                continue
            save_model('last')
            temporary = out / 'training_state.partial.pt'
            torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                        'best_metric': best, 'best_step': best_step, 'stale_evaluations': stale, 'config': cfg,
                        'last_weights_sha256': sha(out / 'last/model.safetensors')}, temporary)
            temporary.replace(out / 'training_state.pt')
            if step >= cfg['minimum_steps'] and stale >= cfg['patience_evaluations']:
                emit(out, {'stage': 'early_stop', 'step': step, 'best_step': best_step})
                break
    delta = float((first_param.detach() - initial).norm())
    assert delta > 0 and math.isfinite(delta)
    emit(out, {'stage': 'smoke_completed' if smoke else 'completed', 'steps': step,
               'action_out_weight_delta_l2': delta, 'best_step': best_step, 'best_metric': None if smoke else best,
               'quantization_started': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/so100_real_finetune_v1.json'))
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'train'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.resume and args.stage != 'train':
        parser.error('--resume is only available for training')
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare' and cfg.get('evaluation') == 'scripted_sim':
        from qvla.evaluation.finetune_so101_scripted import prepare_scripted
        prepare_scripted(cfg, out)
    elif args.stage == 'prepare':
        prepare(cfg, out)
    else:
        train(cfg, out, args.stage == 'smoke', resume=args.resume)


if __name__ == '__main__':
    main()
