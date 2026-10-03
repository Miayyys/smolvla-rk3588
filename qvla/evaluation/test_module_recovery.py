
# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import unittest
import torch
from qvla.haq.module_recovery import recovery_group, recover_state


class RecoveryTests(unittest.TestCase):
    def test_disjoint_groups_and_exact_expert_boundaries(self):
        p='model.vlm_with_expert.lm_expert.layers.'
        self.assertEqual(recovery_group(p+'7.self_attn.q_proj.weight'),'expert_first8')
        self.assertEqual(recovery_group(p+'8.mlp.up_proj.weight'),'expert_last8')
        self.assertEqual(recovery_group('model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight'),'language')
        self.assertEqual(recovery_group('model.state_proj.bias'),'action_interface')
        self.assertIsNone(recovery_group('model.vlm_with_expert.vlm.lm_head.weight'))

    def test_recovery_only_changes_selected_weights_and_keeps_master_dtype(self):
        other=torch.tensor([3.])
        state={'x.master_weight':torch.tensor([2.]),'x.bias':torch.tensor([4.]),'other':other}
        source={'x.weight':torch.tensor([1.],dtype=torch.bfloat16),'x.bias':torch.tensor([0.])}
        report=recover_state(state,source,{'x'})
        self.assertEqual(report['changed_tensors'],2)
        self.assertEqual(state['x.master_weight'].dtype,torch.float32)
        self.assertEqual(state['x.master_weight'].item(),1.)
        self.assertIs(state['other'],other)
        with self.assertRaisesRegex(ValueError,'mismatch'):
            recover_state(state,{'x.weight':torch.zeros(2)},{'x'})


if __name__=='__main__':unittest.main()
