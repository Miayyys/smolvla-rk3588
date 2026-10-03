import unittest
import torch
from torch import nn
from qvla_haq.training_scope import apply_training_scope, frozen_parameter_digest


class Wrapped(nn.Module):
    def __init__(self):
        super().__init__();self.master_weight=nn.Parameter(torch.ones(2,2));self.bias=nn.Parameter(torch.ones(2))


class ScopeTests(unittest.TestCase):
    def test_only_first_eight_layers_update_and_frozen_digest_is_exact(self):
        policy=nn.Module();policy.model=nn.Module();policy.model.vlm_with_expert=nn.Module()
        policy.model.vlm_with_expert.lm_expert=nn.Module()
        policy.model.vlm_with_expert.lm_expert.layers=nn.ModuleList([Wrapped() for _ in range(16)])
        policy.model.action_out_proj=Wrapped()
        audit=apply_training_scope(policy,'expert_first8')
        self.assertEqual(audit['trainable_parameter_elements'],48)
        self.assertEqual(len(audit['trainable_names']),16)
        before=frozen_parameter_digest(policy)
        optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=.01)
        loss=sum(p.sum() for p in policy.parameters() if p.requires_grad);loss.backward();optimizer.step()
        self.assertEqual(before,frozen_parameter_digest(policy))
        self.assertIsNone(policy.model.action_out_proj.master_weight.grad)
        with torch.no_grad():policy.model.action_out_proj.master_weight[0,0]+=1
        self.assertNotEqual(before,frozen_parameter_digest(policy))

    def test_expert_and_interface_excludes_language_and_norms(self):
        policy=nn.Module();policy.model=nn.Module();policy.model.vlm_with_expert=nn.Module()
        policy.model.vlm_with_expert.lm_expert=nn.Module()
        policy.model.vlm_with_expert.lm_expert.layers=nn.ModuleList([Wrapped() for _ in range(16)])
        policy.model.vlm_with_expert.lm_expert.layers[0].norm=nn.LayerNorm(2)
        policy.model.vlm_with_expert.vlm=Wrapped()
        policy.model.action_out_proj=Wrapped()
        audit=apply_training_scope(policy,'expert_and_interface')
        self.assertEqual(audit['trainable_parameter_elements'],102)
        self.assertFalse(policy.model.vlm_with_expert.vlm.master_weight.requires_grad)
        self.assertFalse(policy.model.vlm_with_expert.lm_expert.layers[0].norm.weight.requires_grad)
        self.assertTrue(policy.model.action_out_proj.master_weight.requires_grad)


if __name__=='__main__':unittest.main()
