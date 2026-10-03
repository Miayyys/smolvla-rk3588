"""Disjoint source-weight recovery groups for FP regression interventions."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


GROUPS = ('vision', 'language', 'connector', 'expert_first8', 'expert_last8', 'action_interface')


def recovery_group(key):
    if key.startswith('model.vlm_with_expert.vlm.model.vision_model.'):
        return 'vision'
    if key.startswith('model.vlm_with_expert.vlm.model.text_model.'):
        return 'language'
    if key.startswith('model.vlm_with_expert.vlm.model.connector.'):
        return 'connector'
    prefix = 'model.vlm_with_expert.lm_expert.layers.'
    if key.startswith(prefix):
        layer = int(key[len(prefix):].split('.')[0])
        if not 0 <= layer < 16:
            raise ValueError('Unexpected expert depth')
        return 'expert_first8' if layer < 8 else 'expert_last8'
    if any(key.startswith('model.'+name+'.') for name in
           ('state_proj', 'action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out')):
        return 'action_interface'
    return None


def recover_state(trained, source, wrapped_names):
    """Replace only selected source entries; leave every other tensor untouched."""
    mapping = {}
    for key, value in source.items():
        module, leaf = key.rsplit('.', 1)
        destination = module+'.master_weight' if module in wrapped_names and leaf == 'weight' else key
        if destination not in trained or trained[destination].shape != value.shape:
            raise ValueError('Recovery state mismatch: '+key)
        mapping[key] = destination
    changed = sum(not trained[destination].float().equal(source[key].float())
                  for key, destination in mapping.items())
    for key, destination in mapping.items():
        trained[destination] = source[key].to(dtype=trained[destination].dtype)
    return dict(source_keys=list(mapping), destination_keys=list(mapping.values()),
                restored_tensors=len(mapping), changed_tensors=changed,
                restored_parameter_elements=sum(v.numel() for v in source.values()))
