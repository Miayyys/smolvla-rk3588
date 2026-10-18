"""Goal-conditioned SmolVLA for SO101 (experiments B/C/D in docs/so101-goal-conditioned-design.md).

Two modules are attached to an unmodified SmolVLA policy:

* ``goal_embedding``: a 20-D goal vector (phase one-hot 9, delta position 3, relative rotation 6D 6,
  target gripper aperture 1, delta distance 1) -> MLP whose last layer is zero-initialised. The result is
  added to every action token of the Expert suffix. With the zero init the policy is bit-identical to the
  original at step 0, and the prefix, the suffix length, attention masks and position ids are unchanged.
* ``goal_planner``: reads the frozen VLM's final prefix hidden states (detached) through learned-query
  cross-attention (slow part) plus the latest normalised joint state (fast MLP) and predicts phase logits,
  goal TCP position, goal rotation (6D) and gripper aperture. Its parameters and gradients are disjoint
  from the Expert's.

Frames: positions in metres in the world frame, which equals the robot base frame in this scene (base body
at the origin with identity orientation). Rotations are 3x3 matrices; ``goal_rot`` is the row-major TCP goal
rotation recorded by the generator, ``tcp_pose`` is [x, y, z, qw, qx, qy, qz] from forward kinematics.
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PHASES = ['pregrasp', 'descend', 'close', 'lift', 'carry', 'lower', 'open', 'retreat', 'hold']
# Final gripper command of each expert phase in dataset units (0-100): GRIPPER_OPEN 0.9 rad, GRIPPER_CLOSED
# -0.17 rad and release_open 0.55 rad converted with sim_qpos_to_dataset_row and the env gripper limits
# (-0.17453, 1.74533) rad. The aperture is therefore fixed by the phase in this expert.
APERTURE = {'pregrasp': 55.9692, 'descend': 55.9692, 'close': 0.2360, 'lift': 0.2360, 'carry': 0.2360,
            'lower': 0.2360, 'open': 37.7387, 'retreat': 37.7387, 'hold': 37.7387}
GOAL_DIM = 20
POS_SCALE = 0.1  # m; delta positions and planner positions are expressed in units of 10 cm
STATE_DIM = 6


def quat_to_matrix(q):
    """[w, x, y, z] (..., 4) -> rotation matrices (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_6d(R):
    """First two columns, concatenated: (..., 3, 3) -> (..., 6)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def matrix_from_6d(x):
    """Gram-Schmidt on the two predicted columns: (..., 6) -> (..., 3, 3)."""
    a = F.normalize(x[..., :3], dim=-1)
    b = F.normalize(x[..., 3:] - (a * x[..., 3:]).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def geodesic_deg(R1, R2):
    cos = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)))


def axis_angle_matrix(axis, angle):
    """Rodrigues: unit axes (..., 3), angles in radians (...) -> (..., 3, 3)."""
    K = torch.zeros(*axis.shape[:-1], 3, 3, dtype=axis.dtype, device=axis.device)
    K[..., 0, 1], K[..., 0, 2] = -axis[..., 2], axis[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axis[..., 2], -axis[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axis[..., 1], axis[..., 0]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    return eye + s * K + (1 - c) * (K @ K)


def sample_goal_noise(batch, pos_sigma_m, rot_sigma_deg, device, generator=None):
    """Isotropic position offset (per-axis sigma = pos_sigma_m / sqrt(3), so the RMS norm is pos_sigma_m) and a
    rotation about a uniformly random axis with angle ~ N(0, rot_sigma_deg). Returns (offset (B,3), R (B,3,3))."""
    offset = torch.randn(batch, 3, device=device, generator=generator) * (pos_sigma_m / math.sqrt(3))
    axis = F.normalize(torch.randn(batch, 3, device=device, generator=generator), dim=-1)
    angle = torch.randn(batch, device=device, generator=generator) * math.radians(rot_sigma_deg)
    return offset, axis_angle_matrix(axis, angle)


def goal_vector(phase, goal_pos, goal_rot, tcp_pos, tcp_rot, pos_offset=None, rot_noise=None):
    """20-D goal vector from the gated phase (B,) int, goal (B,3)/(B,3,3) and the current TCP (B,3)/(B,3,3).

    Optional noise perturbs the goal (world frame) before the deltas are formed. The relative rotation is
    expressed in the current TCP frame: R_rel = R_tcp^T R_goal.
    """
    if pos_offset is not None:
        goal_pos = goal_pos + pos_offset
    if rot_noise is not None:
        goal_rot = rot_noise @ goal_rot
    delta = goal_pos - tcp_pos
    relative = tcp_rot.transpose(-1, -2) @ goal_rot
    aperture = torch.tensor([APERTURE[p] for p in PHASES], dtype=goal_pos.dtype, device=goal_pos.device)[phase]
    return torch.cat([F.one_hot(phase, len(PHASES)).to(goal_pos.dtype), delta / POS_SCALE, matrix_to_6d(relative),
                      (aperture / 100)[:, None], delta.norm(dim=-1, keepdim=True) / POS_SCALE], -1)


def goal_vector_from_labels(goal, pos_offset=None, rot_noise=None):
    """Goal vector from a batch of recorded labels: ``goal`` holds phase, goal_pos, goal_rot (row-major 9), tcp_pose."""
    B = goal['phase'].shape[0]
    return goal_vector(goal['phase'], goal['goal_pos'], goal['goal_rot'].reshape(B, 3, 3),
                       goal['tcp_pose'][:, :3], quat_to_matrix(goal['tcp_pose'][:, 3:]), pos_offset, rot_noise)


class GoalEmbedding(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.inp = nn.Linear(GOAL_DIM, hidden)
        self.out = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.null = nn.Parameter(torch.zeros(hidden))  # learned "no goal" embedding, also zero at init
        self.current = None  # (B, hidden) embedding added to the action tokens of the next suffix embeds

    def forward(self, vector, null_mask=None):
        embedding = self.out(F.silu(self.inp(vector)))
        if null_mask is not None:
            embedding = torch.where(null_mask[:, None], self.null.to(embedding.dtype).expand_as(embedding), embedding)
        return embedding


class GoalPlanner(nn.Module):
    def __init__(self, vlm_hidden, queries=2, width=256, heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, vlm_hidden) * 0.02)
        self.attention = nn.MultiheadAttention(vlm_hidden, heads, batch_first=True)
        self.slow = nn.Sequential(nn.LayerNorm(queries * vlm_hidden), nn.Linear(queries * vlm_hidden, width))
        self.fast = nn.Sequential(nn.Linear(width + STATE_DIM, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, len(PHASES) + 3 + 6 + 1)

    def slow_features(self, prefix_out, prefix_pad_masks):
        """Once per VLM call: pooled prefix features (B, width)."""
        queries = self.queries[None].expand(prefix_out.shape[0], -1, -1).to(prefix_out.dtype)
        pooled, _ = self.attention(queries, prefix_out, prefix_out, key_padding_mask=~prefix_pad_masks, need_weights=False)
        return self.slow(pooled.flatten(1))

    def forward(self, features, state):
        """Per control step: features (B, width) + normalised joint state (B, 6)."""
        out = self.head(self.fast(torch.cat([features, state.to(features.dtype)], -1))).float()
        n = len(PHASES)
        return {'phase_logits': out[:, :n], 'goal_pos': out[:, n:n + 3] * POS_SCALE,
                'goal_rot': matrix_from_6d(out[:, n + 3:n + 9]), 'aperture': out[:, n + 9] * 100}


def attach(policy):
    """Add goal_embedding/goal_planner to ``policy.model`` (so they are in its state_dict) and route the
    Expert suffix through the goal embedding. Idempotent per policy object only: call once."""
    model = policy.model
    if hasattr(model, 'goal_embedding'):
        raise ValueError('Goal modules already attached')
    device = next(model.parameters()).device
    model.goal_embedding = GoalEmbedding(model.vlm_with_expert.expert_hidden_size).to(device)
    model.goal_planner = GoalPlanner(model.vlm_with_expert.config.text_config.hidden_size).to(device)
    original = model.embed_suffix

    def embed_suffix(noisy_actions, timestep):
        embs, pad_masks, att_masks = original(noisy_actions, timestep)
        current = model.goal_embedding.current
        if current is not None:
            embs = embs + current[:, None, :].to(embs.dtype)
        return embs, pad_masks, att_masks

    model.embed_suffix = embed_suffix
    return policy


def planner_parameter(name):
    return '.goal_planner.' in f'.{name}'


def phase_weights(phases):
    """Inverse-frequency class weights over the training frames, normalised to mean 1 over the 9 phases."""
    counts = np.bincount(np.asarray(phases), minlength=len(PHASES)).astype('float64')
    weights = counts.sum() / np.maximum(counts, 1) / len(PHASES)
    return weights / weights.mean()


def _inputs(policy, batch):
    from lerobot.policies.smolvla import modeling_smolvla as ms
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return images, img_masks, batch[ms.OBS_LANGUAGE_TOKENS], batch[ms.OBS_LANGUAGE_ATTENTION_MASK], state


def prefix_features(policy, batch):
    """Final-layer prefix hidden states of the frozen VLM (prefix tokens never attend to the suffix).

    Prefix-only calls must take the cache-prefill path (``use_cache=True`` without a cache), as
    ``sample_actions`` does: every layer then runs self-attention over the prefix."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    model = policy.model
    images, img_masks, lang_tokens, lang_masks, state = _inputs(policy, batch)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    (prefix_out, _), _ = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
        position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True)
    return prefix_out, prefix_pad_masks, state[:, :STATE_DIM]


def planner_loss(out, goal, class_weights):
    """Weighted CE (phase) + Huber on position in cm (delta 1 cm) + geodesic angle in units of 10 deg + L1 on
    aperture in units of 10 gripper points. Each term is O(1) for the errors that matter here."""
    phase = goal['phase']
    target_rot = goal['goal_rot'].reshape(-1, 3, 3).float()
    aperture = torch.tensor([APERTURE[p] for p in PHASES], device=phase.device)[phase]
    terms = {'phase_ce': F.cross_entropy(out['phase_logits'], phase, weight=class_weights),
             'position_huber_cm': F.huber_loss(out['goal_pos'] * 100, goal['goal_pos'].float() * 100, delta=1.0),
             'rotation_deg_over_10': geodesic_deg(out['goal_rot'], target_rot).mean() / 10,
             'aperture_l1_over_10': (out['aperture'] - aperture).abs().mean() / 10}
    return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}


def planner_metrics(out, goal):
    phase = goal['phase']
    aperture = torch.tensor([APERTURE[p] for p in PHASES], device=phase.device)[phase]
    return {'phase_correct': (out['phase_logits'].argmax(-1) == phase).float().cpu().numpy(),
            'position_error_mm': ((out['goal_pos'] - goal['goal_pos'].float()).norm(dim=-1) * 1000).cpu().numpy(),
            'rotation_error_deg': geodesic_deg(out['goal_rot'], goal['goal_rot'].reshape(-1, 3, 3).float()).cpu().numpy(),
            'aperture_error': (out['aperture'] - aperture).abs().cpu().numpy(),
            'phase': phase.cpu().numpy()}


def training_loss(policy, batch, goal, goal_cfg, class_weights, noise=None, time=None, return_prefix=False):
    """Flow-matching loss of the goal-conditioned Expert (same masking/reduction as SmolVLAPolicy.forward) and
    the planner loss on detached prefix features. Goal noise and null-goal dropout follow ``goal_cfg``."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    model = policy.model
    images, img_masks, lang_tokens, lang_masks, state = _inputs(policy, batch)
    actions = policy.prepare_action(batch)
    if noise is None:
        noise = model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = model.sample_time(actions.shape[0], actions.device)
    B = actions.shape[0]
    offset, rot = sample_goal_noise(B, goal_cfg['goal_position_noise_m'], goal_cfg['goal_rotation_noise_deg'], actions.device)
    vector = goal_vector_from_labels(goal, offset, rot)
    null_mask = torch.rand(B, device=actions.device) < goal_cfg['null_goal_probability']
    model.goal_embedding.current = model.goal_embedding(vector, null_mask)
    try:
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        (prefix_out, suffix_out), _ = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad_masks, att_masks), position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
    finally:
        model.goal_embedding.current = None
    v_t = model.action_out_proj(suffix_out[:, -policy.config.chunk_size:].to(torch.float32))
    losses = F.mse_loss(u_t, v_t, reduction='none')[:, :, :policy.config.action_feature.shape[0]]
    valid = ~batch['action_is_pad']
    losses = losses * valid.unsqueeze(-1)
    flow = losses.sum() / (valid.sum() * losses.shape[-1]).clamp_min(1)
    features = model.goal_planner.slow_features(prefix_out.detach(), prefix_pad_masks)
    out = model.goal_planner(features, state[:, :STATE_DIM])
    planner, terms = planner_loss(out, goal, class_weights)
    if return_prefix:
        return flow, planner, terms, prefix_out
    return flow, planner, terms


def set_goal(policy, vector, null_mask=None):
    """Condition the next predict_action_chunk call(s) on ``vector`` (B, 20); ``None`` clears it."""
    model = policy.model
    model.goal_embedding.current = None if vector is None else model.goal_embedding(vector, null_mask)


def labels_to_device(goal, device):
    return {k: v.to(device, non_blocking=True) for k, v in goal.items()}


def load_goal_policy(model_dir, vlm_assets_dir, device='cuda'):
    """Load a checkpoint saved by the trainer with goal modules (strict: every tensor must match)."""
    from pathlib import Path
    from lerobot.policies.smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from safetensors.torch import load_file
    config = SmolVLAConfig.from_pretrained(model_dir)
    config.device = device
    config.vlm_model_name = str(Path(vlm_assets_dir).resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy(config)
    attach(policy)
    policy.load_state_dict(load_file(str(Path(model_dir) / 'model.safetensors'), device='cpu'), strict=True)
    return policy.to(device).eval()
