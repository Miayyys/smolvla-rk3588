#!/usr/bin/env python3
"""Cache/metric regression tests; no full SmolVLA or LIBERO rollout is run."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qvla_haq.offline_actions import (action_seed, frame_positions, load_cache,
                                      save_cache, score_actions, validate_partition)


class OfflineActionsTests(unittest.TestCase):
    def test_partition_leakage(self):
        p = {'development_episode_ids_from_qat_train': [1],
             'calibration_episode_ids_from_ptq_calibration': [2],
             'remaining_frozen_test_episode_ids': [3],
             'exploratory_test_episode_ids_to_exclude': [4]}
        s = {'splits': {'qat_train': [1], 'ptq_calibration': [2], 'test': [3, 4]}}
        self.assertEqual(validate_partition(p, s), [1])
        p['remaining_frozen_test_episode_ids'] = [1, 3]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_partition(p, s)

    def test_frame_and_noise_identity(self):
        self.assertEqual(frame_positions(101, 0, 3), [0, 50, 90])
        for task in range(40):
            frame = frame_positions(101, task, 1)[0]
            self.assertIn(frame, frame_positions(101, task, 3))
            self.assertEqual(action_seed(1, task, frame), action_seed(1, task, frame))
        with self.assertRaises(ValueError):
            frame_positions(2, 0, 3)

    def test_cache_identity_and_corruption(self):
        observations = {'image1': np.zeros((2, 3, 4, 4), dtype=np.uint8),
                        'image2': np.zeros((2, 3, 4, 4), dtype=np.uint8),
                        'state': np.zeros((2, 6), dtype=np.float32),
                        'task': np.array(['one', 'two']),
                        'recorded_action': np.zeros((2, 7), dtype=np.float32)}
        actions = np.zeros((2, 50, 7), dtype=np.float32)
        manifest = {'identity': {'checkpoint': 'test_only'}, 'samples': [{}, {}]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'cache'
            save_cache(directory, manifest, observations, actions)
            _, actual, fp = load_cache(directory, manifest['identity'])
            np.testing.assert_array_equal(fp, actions)
            np.testing.assert_array_equal(actual['task'], observations['task'])
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_cache(directory, {'checkpoint': 'different'})
            with self.assertRaises(FileExistsError):
                save_cache(directory, manifest, observations, actions)
            with (directory / 'observations.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_cache(directory)

    def test_task_balance_and_gripper(self):
        reference = np.ones((3, 2, 7), dtype=np.float32)
        candidate = reference.copy()
        candidate[0] += 1
        candidate[1:] += 3
        scored = score_actions(candidate, reference, reference[:, 0], np.array([0, 1, 1]))
        self.assertEqual(scored['task_macro']['chunk_mae_vs_fp'], 2)
        candidate = reference.copy()
        candidate[:, :, 6] = -1
        scored = score_actions(candidate, reference, reference[:, 0], np.array([0, 1, 2]))
        self.assertEqual(scored['task_macro']['continuous_chunk_mae_vs_fp'], 0)
        self.assertEqual(scored['task_macro']['gripper_sign_disagreement_vs_fp'], 1)

    def test_bad_candidate_rejected(self):
        good = np.zeros((2, 50, 7), dtype=np.float32)
        for bad in (good[:, :49], np.full_like(good, np.nan)):
            with self.assertRaises(ValueError):
                score_actions(bad, good, good[:, 0], np.array([0, 1]))

    def test_archived_real_model_metric_regression(self):
        archive = ROOT / 'runs/expert_real_w8a8_action_v1/actions.npz'
        report_path = archive.parent / 'report.json'
        if not archive.exists() or not report_path.exists():
            self.skipTest('Archived real-model action evidence not available on this host')
        report = json.loads(report_path.read_text())
        with np.load(archive, allow_pickle=False) as stored:
            for mode in ('ptq', 'qat'):
                scores = score_actions(stored[mode], stored['original'],
                                       stored['recorded_first'], stored['task_index'])
                self.assertAlmostEqual(scores['task_macro']['chunk_mae_vs_fp'],
                                       report[mode]['metrics']['chunk_mae_vs_original_fp'], places=7)
                self.assertAlmostEqual(scores['task_macro']['first_action_mae_vs_recorded'],
                                       report[mode]['metrics']['first_action_mae_vs_recorded'], places=7)
            zero = score_actions(stored['original'], stored['original'],
                                 stored['recorded_first'], stored['task_index'])
            self.assertEqual(zero['task_macro']['chunk_mae_vs_fp'], 0)

    def paired_runner(self, device):
        import torch
        sys.path.insert(0, str(ROOT / 'scripts'))
        from haq_offline_eval import predict_cached

        class SmallPolicy:
            def reset(self):
                pass

            def predict_action_chunk(self, observation):
                return torch.rand(1, 50, 7, device=device) + observation['observation.state'].to(device).sum()

        observations = {'image1': np.zeros((1, 3, 4, 4), dtype=np.uint8),
                        'image2': np.zeros((1, 3, 4, 4), dtype=np.uint8),
                        'state': np.ones((1, 6), dtype=np.float32), 'task': np.array(['test'])}
        sample = {'episode_index': 1, 'task_index': 2, 'frame_index': 3,
                  'action_seed': action_seed(1, 2, 3)}
        state = torch.random.get_rng_state().clone()
        gpu_state = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
        a, _ = predict_cached(SmallPolicy(), lambda x: x, lambda x: x, observations, [sample], device)
        b, _ = predict_cached(SmallPolicy(), lambda x: x, lambda x: x, observations, [sample], device)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        if gpu_state is not None:
            self.assertTrue(torch.equal(gpu_state, torch.cuda.get_rng_state()))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable in this Python')
    def test_paired_cpu_runner_restores_rng(self):
        self.paired_runner('cpu')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable in this Python')
    def test_paired_cuda_runner_restores_rng(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA device unavailable in this execution environment')
        self.paired_runner('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
