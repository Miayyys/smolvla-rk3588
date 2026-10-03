"""Explicit trainable parameter scopes applied after mixed operators are built."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import hashlib
import torch
from qvla.haq.module_recovery import recovery_group


def apply_training_scope(policy, scope):
    if scope not in ('all_sites', 'expert_first8', 'expert_and_interface'):
        raise ValueError('Unknown training scope')
    if scope != 'all_sites':
        selected = {'expert_first8'} if scope == 'expert_first8' else {'expert_first8','expert_last8','action_interface'}
        wrapped={n.rsplit('.',1)[0] for n,p in policy.named_parameters() if n.endswith('.master_weight')}
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(recovery_group(name) in selected and
                                     name.rsplit('.',1)[0] in wrapped and
                                     name.endswith(('.master_weight', '.bias')))
    trainable = [(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError('Empty training scope')
    return dict(scope=scope, trainable_names=[n for n,p in trainable],
                trainable_parameter_elements=sum(p.numel() for n,p in trainable),
                frozen_parameter_elements=sum(p.numel() for p in policy.parameters() if not p.requires_grad))


def frozen_parameter_digest(policy):
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
            digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
