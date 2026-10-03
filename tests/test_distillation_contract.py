import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from qvla_haq.distillation import TeacherCache, teacher_batch, combined_flow_loss
from qvla_haq.offline_actions import file_sha256


class ContractTests(unittest.TestCase):
    def test_teacher_cannot_leak_into_evaluation_or_silently_be_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'actions.npy',np.ones((1,8,7),np.float32))
            m=dict(partition_sha256='p',split_sha256='s',source_kind='synthetic_test_only',
                fps=10,action_space='libero_simulator_7d',actions_sha256=file_sha256(root/'actions.npy'),
                rows=[dict(episode_index=3,frame_index=4,task_index=0,suite='libero_10',valid_length=8)])
            (root/'manifest.json').write_text(json.dumps(m))
            kw=dict(partition_sha256='p',split_sha256='s',allowed_episodes=[3])
            with self.assertRaisesRegex(ValueError,'Synthetic'):TeacherCache(root,**kw)
            with self.assertRaisesRegex(ValueError,'excluded'):TeacherCache(root,**{**kw,'allowed_episodes':[5]},allow_synthetic=True)
            c=TeacherCache(root,**kw,allow_synthetic=True)
            self.assertEqual(c.choose(0,random.Random(0)),0)
            with (root/'actions.npy').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):TeacherCache(root,**kw,allow_synthetic=True)

    def test_horizon_mask_preserves_padding_context_and_original_observation(self):
        actions=torch.randn(50,7);pad=torch.zeros(50,dtype=torch.bool);pad[6:]=True
        obs={'action':actions,'action_is_pad':pad}
        # A representative affine normalization: teacher targets must pass through it.
        def pre(b):return {**b,'action':(b['action']-2)/4}
        b=teacher_batch(obs,pre,np.ones((8,7),np.float32),8)
        self.assertEqual(int((~b['action_is_pad']).sum()),6)
        self.assertTrue(torch.equal(b['action'][0,8:],(actions[8:]-2)/4))
        self.assertTrue(torch.equal(obs['action'],actions))
        self.assertTrue(torch.all(b['action'][0,:8]==-.25))

    def test_both_losses_share_flow_noise_time_and_reach_student(self):
        class Model:
            def sample_noise(self,shape,device):return torch.randn(shape,device=device)
            def sample_time(self,n,device):return torch.rand(n,device=device)
        class Policy:
            def __init__(self):self.model=Model();self.w=torch.tensor(.5,requires_grad=True);self.calls=[]
            def prepare_action(self,b):return b['action']
            def forward(self,b,noise,time):
                self.calls.append((noise,time))
                return ((self.w-b['action'])**2).mean(),{}
        p=Policy();loss,parts=combined_flow_loss(p,{'action':torch.ones(1,50,7)},
            {'action':torch.zeros(1,50,7)},.2)
        self.assertIs(p.calls[0][0],p.calls[1][0]);self.assertIs(p.calls[0][1],p.calls[1][1])
        loss.backward();self.assertAlmostEqual(float(p.w.grad),-.8,places=5)
        self.assertAlmostEqual(parts['teacher_loss'],.25)
        control=Policy()
        gt,parts=combined_flow_loss(control,{'action':torch.ones(1,50,7)},
            {'action':torch.zeros(1,50,7)},0.)
        gt.backward()
        self.assertEqual(len(control.calls),1)
        self.assertIsNone(parts['teacher_loss'])
        self.assertAlmostEqual(float(control.w.grad),-1.,places=5)

if __name__=='__main__':unittest.main()
