"""Reconstruct FP state from full masters or exact trainable tensor overlays.

An overlay is stored floating weights, not LoRA and not a quantized artifact.
The caller verifies source/overlay file hashes before loading.
"""


def source_key(key):
    return key.removesuffix('.master_weight')+'.weight' if key.endswith('.master_weight') else key


def reconstruct_fp_state(saved, base, wrapped_names, report):
    if report.get('master_storage_format', 'full_fp_master') == 'trainable_fp_overlay':
        scope = report['training_scope']
        if not scope['frozen_parameters_unchanged']:
            raise ValueError('Overlay source frozen parameters were modified')
        expected = {source_key(k) for k in scope['trainable_names']}
        if set(saved) != expected:
            raise ValueError('Overlay does not exactly cover declared trainable tensors')
        state = dict(base)
    elif report.get('master_storage_format', 'full_fp_master') == 'full_fp_master':
        state = {}
    else:
        raise ValueError('Unknown FP master storage format')
    for key, value in saved.items():
        module, leaf = key.rsplit('.', 1)
        destination = module+'.master_weight' if module in wrapped_names and leaf == 'weight' else key
        if destination not in base or base[destination].shape != value.shape:
            raise ValueError('FP state shape/key mismatch: '+key)
        state[destination] = value
    if set(state) != set(base):
        raise ValueError('Incomplete reconstructed FP state')
    return state
