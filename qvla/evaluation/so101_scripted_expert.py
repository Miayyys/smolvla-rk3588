"""Privileged scripted pick-and-place expert for SO101-Nexus MuJoCoPickAndPlace-v1.

The expert reads simulator ground truth (object/target pose) to compute joint
targets; the recorded policy inputs remain only RGB images and the 6-D joint
state. Each control step re-solves IK from the measured joint state toward the
next Cartesian reference point, so the expert's label is a feedback action:
with injected execution noise (DART-style) the clean label is the correction
back toward the plan. Phases advance on measured progress, not wall time, and a
failed grasp triggers a bounded retry, which records recovery behaviour.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NEXUS_SOURCE = '/home/loser/Study/so101-nexus'
PROFILE = 'artifacts/datasets/so101_sim_pickplace/meta/so101_nexus_env.json'
GRIPPER_OPEN = 0.9      # rad; jaw clears a 25 mm cube
GRIPPER_CLOSED = -0.17  # rad; lower joint target, contact stops the jaw first
# Coarse transport phases get execution noise; fine alignment and jaw actions stay clean.
NOISY_PHASES = ('pregrasp', 'lift', 'carry')


def make_env(image_size=(320, 240), cameras=True, max_episode_steps=1200):
    sys.path.insert(0, str(Path(NEXUS_SOURCE) / 'src'))
    import gymnasium as gym
    import so101_nexus.mujoco  # noqa: F401
    from so101_nexus import JointPositions, OverheadCamera, PhysicsConfig, PickAndPlaceConfig, WristCamera
    from so101_nexus.teleop.config_customization import apply_config_overrides, load_profile_overrides
    w, h = image_size
    observations = [JointPositions()]
    if cameras:
        observations += [OverheadCamera(width=w, height=h), WristCamera(width=w, height=h)]
    cfg = PickAndPlaceConfig(obs_mode='visual' if cameras else 'state', observations=observations,
        physics=PhysicsConfig(timestep_s=1/300, control_period_s=1/30,
                              integrator='implicitfast', noslip_iterations=3, tolerance=1e-8))
    cfg = apply_config_overrides(cfg, load_profile_overrides(Path(PROFILE), 'MuJoCoPickAndPlace-v1', cfg))
    return gym.make('MuJoCoPickAndPlace-v1', config=cfg, control_mode='pd_joint_pos',
                    max_episode_steps=max_episode_steps)


def info_grasped(expert):
    return expert.u._is_grasping() >= 0.5


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def grasp_frame(obj_xy, yaw, tilt):
    """TCP rotation: z = approach (tilted outward in the pan plane), x = jaw closing axis.

    A 5-joint arm keeps its approach axis in the vertical plane through the pan
    axis, so tilt only rotates toward the radial direction. wrist_roll then sets
    the closing axis; it is chosen so its horizontal projection is the cube face
    normal nearest to the tangential direction.
    """
    radial = _unit(np.array([obj_xy[0], obj_xy[1], 0.0]))
    up = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(up, radial)
    a = np.sin(tilt) * radial - np.cos(tilt) * up
    b = _unit(np.cross(a, tangent))
    # cube face normal direction relative to tangent, folded to [-45, 45] deg
    phi = (yaw - np.arctan2(tangent[1], tangent[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    horizontal_b = b @ radial  # |cos(tilt)|
    psi = np.arctan2(np.sin(phi), np.cos(phi) * horizontal_b) if abs(horizontal_b) > 1e-6 else phi
    # radial is tangent rotated -90 deg, so a positive face angle needs -b
    c = np.cos(psi) * tangent - np.sin(psi) * b * np.sign(horizontal_b)
    c = _unit(c - (c @ a) * a)
    y = np.cross(a, c)
    return np.stack([c, y, a], axis=1)


class IK:
    """Damped least squares on a scratch MjData; position-weighted, orientation soft."""

    def __init__(self, env, orientation_weight=0.15, damping=0.02, iterations=40):
        import mujoco
        self.mj = mujoco
        self.u = env.unwrapped
        self.scratch = mujoco.MjData(self.u.model)
        self.w, self.damping, self.iterations = orientation_weight, damping, iterations
        self.low, self.high = self.u._target_low, self.u._target_high

    def solve(self, q0, pos, rot, fixed_roll=None):
        """Joint targets for a TCP pose; with ``fixed_roll`` only the approach axis is constrained."""
        mj, u, d = self.mj, self.u, self.scratch
        d.qpos[:] = u.data.qpos
        q = q0[:5].copy()
        if fixed_roll is not None:
            q[4] = fixed_roll
        n = 4 if fixed_roll is not None else 5
        jacp = np.zeros((3, u.model.nv)); jacr = np.zeros((3, u.model.nv))
        dofs = u._arm_qvel_addrs[:n]
        for _ in range(self.iterations):
            d.qpos[u._arm_qpos_addrs[:5]] = q
            mj.mj_kinematics(u.model, d); mj.mj_comPos(u.model, d)
            mj.mj_jacSite(u.model, d, jacp, jacr, u._tcp_site_id)
            R = d.site_xmat[u._tcp_site_id].reshape(3, 3)
            ep = pos - d.site_xpos[u._tcp_site_id]
            if fixed_roll is None:
                er = 0.5 * (np.cross(R[:, 0], rot[:, 0]) + np.cross(R[:, 1], rot[:, 1]) + np.cross(R[:, 2], rot[:, 2]))
            else:
                er = np.cross(R[:, 2], rot[:, 2])
            J = np.vstack([jacp[:, dofs], self.w * jacr[:, dofs]])
            e = np.concatenate([ep, self.w * er])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(6), e)
            q[:n] = np.clip(q[:n] + dq, self.low[:n], self.high[:n])
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(dq) < 1e-5:
                break
        d.qpos[u._arm_qpos_addrs[:5]] = q
        mj.mj_kinematics(u.model, d)
        residual = float(np.linalg.norm(pos - d.site_xpos[u._tcp_site_id]))
        return q, residual


@dataclass
class ExpertParams:
    pregrasp_back: float = 0.07     # m back along approach before descending
    grasp_depth: float = 0.012      # m past cube centre along approach
    min_tcp_z: float = 0.004
    table_clearance: float = 0.002  # m, conservative: geom bounding spheres
    lift_height: float = 0.10
    place_clearance: float = 0.003
    speed: float = 0.22             # m/s Cartesian reference speed
    slow_speed: float = 0.08        # m/s for final approach/placement
    close_steps: int = 14
    open_steps: int = 24
    release_open: float = 0.55   # rad; partial opening frees the cube without flicking it
    reach_tol: float = 0.006
    max_attempts: int = 2
    tilt_candidates: tuple = (0.35, 0.55, 0.75, 0.95, 1.15, 1.3)
    max_joint_step_deg: float = 4.0  # per 30 Hz step, arm joints
    stall_steps: int = 6


@dataclass
class Expert:
    env: object
    params: ExpertParams = field(default_factory=ExpertParams)

    def __post_init__(self):
        self.u = self.env.unwrapped
        self.ik = IK(self.env)
        self.reset()

    def reset(self):
        self.phase = 'pregrasp'; self.attempt = 1; self.counter = 0
        self.gripper = GRIPPER_OPEN
        self.ref = self.u._get_tcp_pose()[:3].copy()
        self.place_rot = None; self.lift_from = None; self.failed = False
        self.carry_offset = np.zeros(3); self.retreat_point = self.ref.copy()
        obj = self.u._get_object_pose()
        self.grasp_rot, self.tilt = self._choose_frame(obj[:3], self._yaw(obj))
        self.prev_label = self.u.data.qpos[self.u._arm_qpos_addrs].copy()
        self.prev_tcp = self.ref.copy(); self.stall = 0; self.roll_hold = None
        self.done = False

    @staticmethod
    def _yaw(pose):
        w, x, y, z = pose[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _choose_frame(self, xyz, yaw):
        """Smallest reachable tilt; of the two jaw directions, the one with smaller |wrist_roll|.

        The cube is symmetric under a 180-degree jaw flip, but near the roll
        limits the opened moving jaw can jam against the table.
        """
        q0 = self.u.data.qpos[self.u._arm_qpos_addrs].copy()
        best = None
        for tilt in self.params.tilt_candidates:
            options = []
            for flip in (1.0, -1.0):
                rot = grasp_frame(xyz[:2], yaw, tilt)
                rot = rot * np.array([flip, flip, 1.0])
                target = self._grasp_point(xyz, rot)
                q, res = self.ik.solve(q0, target, rot)
                # Fingertips reach below the TCP: raise the grasp until the hand
                # clears the table with the jaw both open and closed.
                clearance = min(self._jaw_clearance(q, GRIPPER_OPEN), self._jaw_clearance(q, GRIPPER_CLOSED))
                raise_z = max(0.0, self.params.table_clearance - clearance)
                if raise_z > 0:
                    q, res = self.ik.solve(q0, target + np.array([0, 0, raise_z]), rot)
                options.append((res > 1.5e-3, raise_z, res, rot))
            options.sort(key=lambda o: (o[0], o[1]))
            _, raise_z, res, rot = options[0]
            if best is None or res < best[2] - 1e-3:
                best = (rot, tilt, res, raise_z)
            if res < 1.5e-3:
                self.grasp_raise = raise_z
                return rot, tilt
        self.grasp_raise = best[3]
        return best[0], best[1]

    def _jaw_clearance(self, q, gripper):
        """Lowest gripper/jaw collision-geom bound (m) at arm pose ``q`` and the given jaw angle."""
        u, d, mj = self.u, self.ik.scratch, self.ik.mj
        d.qpos[u._arm_qpos_addrs[:5]] = q[:5]
        d.qpos[u._qpos_addrs[5]] = gripper
        mj.mj_kinematics(u.model, d)
        if not hasattr(self, '_hand_geoms'):
            bodies = {mj.mj_name2id(u.model, mj.mjtObj.mjOBJ_BODY, n) for n in ('gripper', 'moving_jaw_so101_v1')}
            self._hand_geoms = [g for g in range(u.model.ngeom)
                                if u.model.geom_bodyid[g] in bodies and u.model.geom_contype[g]]
        return float(min(d.geom_xpos[g][2] - u.model.geom_rbound[g] for g in self._hand_geoms))

    def _grasp_point(self, xyz, rot):
        p = xyz + rot[:, 2] * self.params.grasp_depth
        p[2] = max(p[2], self.params.min_tcp_z) + getattr(self, 'grasp_raise', 0.0)
        return p

    def _goal(self):
        P = self.params; obj = self.u._get_object_pose(); tgt = self.u._get_target_pos()
        rot = self.grasp_rot
        if self.phase == 'pregrasp':
            return self._grasp_point(obj[:3], rot) - rot[:, 2] * P.pregrasp_back, rot, P.speed
        if self.phase in ('descend', 'close'):
            return self._grasp_point(obj[:3], rot), rot, P.slow_speed
        if self.phase == 'lift':
            g = self._grasp_point(obj[:3], rot) if self.lift_from is None else self.lift_from
            return np.array([g[0], g[1], P.lift_height]), rot, P.speed
        if self.phase in ('carry', 'lower'):
            # Aim with the live in-hand offset so slip during transport is corrected.
            tcp = self.u._get_tcp_pose()[:3]
            offset = tcp - obj[:3] if info_grasped(self) else self.carry_offset
            if self.phase == 'carry':
                return np.array([tgt[0] + offset[0], tgt[1] + offset[1], P.lift_height]), self.place_rot, P.speed
            z = self.u._initial_obj_z + P.place_clearance + offset[2]
            xy = np.array([tgt[0] + offset[0], tgt[1] + offset[1]])
            # Descend fully only once horizontally aligned over the disc.
            if np.linalg.norm(xy - tcp[:2]) > 0.008:
                z = max(z + 0.035, min(tcp[2], P.lift_height))
            return np.array([xy[0], xy[1], max(z, P.min_tcp_z)]), self.place_rot, P.slow_speed
        if self.phase == 'open':
            return self.open_point, self.place_rot, P.slow_speed
        if self.phase in ('retreat', 'hold'):
            return self.retreat_point, self.place_rot, P.slow_speed
        raise ValueError(self.phase)

    def _advance(self, info):
        P = self.params; u = self.u
        tcp = u._get_tcp_pose()[:3]
        goal, _, _ = self._goal()
        moved = np.linalg.norm(tcp - self.prev_tcp); self.prev_tcp = tcp.copy()
        at_ref = np.linalg.norm(goal - self.ref) < 1e-3
        self.stall = self.stall + 1 if (at_ref and moved < 5e-4) else 0
        # Contact (table, held cube) or reach limits can leave a steady offset.
        reached = at_ref and (np.linalg.norm(goal - tcp) < P.reach_tol or self.stall >= P.stall_steps)
        self.counter += 1
        if self.phase == 'pregrasp' and (reached or self.counter > 150):
            self.phase, self.counter = 'descend', 0
        elif self.phase == 'descend' and (reached or self.counter > 90):
            self.phase, self.counter = 'close', 0
        elif self.phase == 'close':
            self.gripper = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * min(1.0, self.counter / P.close_steps)
            if self.counter >= P.close_steps + 6:
                self.phase, self.counter, self.lift_from = 'lift', 0, tcp.copy()
        elif self.phase == 'lift' and (reached or self.counter > 90):
            if info['is_grasped'] < 0.5:
                self._retry()
            else:
                obj = u._get_object_pose()[:3]
                # Carry the TCP so the held object, not the TCP, lands on the disc.
                self.carry_offset = np.array([tcp[0] - obj[0], tcp[1] - obj[1], tcp[2] - obj[2]])
                self.carry_offset[2] = max(self.carry_offset[2], 0.0)
                tgt = u._get_target_pos()
                self.place_rot = self._place_rotation(tgt)
                self.phase, self.counter = 'carry', 0
        elif self.phase == 'carry':
            if info['is_grasped'] < 0.5 and self.counter > 10:
                self._retry()
            elif reached or self.counter > 200:
                self.phase, self.counter = 'lower', 0
        elif self.phase == 'lower' and ((reached and np.linalg.norm(goal[:2] - tcp[:2]) < 0.008
                                          and goal[2] < self.u._initial_obj_z + P.place_clearance + 0.03) or self.counter > 150):
            self.phase, self.counter, self.open_point = 'open', 0, tcp.copy()
        elif self.phase == 'open':
            self.gripper = GRIPPER_CLOSED + (P.release_open - GRIPPER_CLOSED) * min(1.0, self.counter / P.open_steps)
            if self.counter >= P.open_steps + 6:
                self.retreat_point = tcp - self.place_rot[:, 2] * 0.05 + np.array([0, 0, 0.03])
                self.phase, self.counter = 'retreat', 0
        elif self.phase == 'retreat' and (reached or self.counter > 60):
            self.phase, self.counter = 'hold', 0
        elif self.phase == 'hold' and self.counter > 30:
            self.done = True

    def _place_rotation(self, tgt):
        """Approach frame over the target; the smallest tilt reachable at lift and place heights.

        wrist_roll is held during transport, so only the approach axis matters here.
        """
        u = self.u; q0 = u.data.qpos[u._arm_qpos_addrs].copy(); roll = float(self.prev_label[4])
        best = None
        for tilt in self.params.tilt_candidates:
            rot = grasp_frame(tgt[:2], 0.0, tilt)
            worst = max(self.ik.solve(q0, np.array([tgt[0], tgt[1], z]), rot, fixed_roll=roll)[1]
                        for z in (self.params.lift_height, u._initial_obj_z + 0.02))
            if best is None or worst < best[1] - 1e-3:
                best = (rot, worst)
            if worst < 2e-3:
                break
        return best[0]

    def _retry(self):
        if self.attempt >= self.params.max_attempts:
            self.done = True; self.failed = True; return
        self.attempt += 1
        self.gripper = GRIPPER_OPEN
        obj = self.u._get_object_pose()
        self.grasp_rot, self.tilt = self._choose_frame(obj[:3], self._yaw(obj))
        self.place_rot = None; self.lift_from = None
        self.phase, self.counter = 'pregrasp', 0

    def act(self, info):
        """Clean expert joint target (6,) for the current measured state."""
        phase = self.phase
        self._advance(info)
        if self.phase != phase:
            self.stall = 0
        goal, rot, speed = self._goal()
        step = speed / 30.0
        delta = goal - self.ref
        dist = np.linalg.norm(delta)
        self.ref = goal.copy() if dist <= step else self.ref + delta / dist * step
        q_now = self.u.data.qpos[self.u._arm_qpos_addrs].copy()
        transport = self.phase in ('carry', 'lower', 'open', 'retreat', 'hold')
        if transport and self.roll_hold is None:
            self.roll_hold = float(self.prev_label[4])
        if not transport:
            self.roll_hold = None
        q, _ = self.ik.solve(q_now, self.ref, rot, fixed_roll=self.roll_hold)
        limit = np.radians(self.params.max_joint_step_deg)
        q = self.prev_label[:5] + np.clip(q - self.prev_label[:5], -limit, limit)
        self.prev_label = np.concatenate([q, [self.gripper]])
        return np.concatenate([q, [self.gripper]]).astype('float64')


def run_episode(env, seed, noise_std_deg=0.0, noise_theta=0.15, max_steps=900, recorder=None, params=None, rng=None):
    """Roll out the expert. Executed = label + OU noise on the 5 arm joints (none while closing)."""
    obs, info0 = env.reset(seed=seed)
    expert = Expert(env, params or ExpertParams())
    expert.failed = False
    rng = rng or np.random.default_rng(seed + 7919)
    u = env.unwrapped
    noise = np.zeros(5); sigma = np.radians(noise_std_deg)
    info = {'is_grasped': 0.0, 'success': False}
    success = grasped = False; steps = 0; first_success = None
    while steps < max_steps and not expert.done:
        label = expert.act(info)
        noise = noise + noise_theta * (-noise) + sigma * np.sqrt(2 * noise_theta) * rng.standard_normal(5)
        executed = label.copy()
        if sigma > 0 and expert.phase in NOISY_PHASES:
            executed[:5] += noise
        executed = np.clip(executed, u._target_low, u._target_high)
        if recorder is not None:
            recorder(obs, label, executed, expert.phase)
        obs, _, terminated, truncated, info = env.step(executed.astype('float32'))
        steps += 1
        success |= bool(info['success']); grasped |= bool(info['is_grasped'] >= 0.5)
        if info['success'] and first_success is None:
            first_success = steps
        if truncated:
            break
    return {'seed': int(seed), 'success': bool(info['success']), 'ever_success': success, 'ever_grasped': grasped,
            'steps': steps, 'attempts': expert.attempt, 'expert_failed': bool(expert.failed),
            'first_success_step': first_success, 'tilt': float(expert.tilt),
            'final_obj_to_target_m': float(info.get('obj_to_target_dist', np.nan)), 'noise_std_deg': noise_std_deg}
