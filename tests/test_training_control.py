import json
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from qvla_haq.training_control import learning_rate_at,save_checkpoint,restore_checkpoint,stratified_frames
from qvla_haq.distillation import TeacherCache,teacher_batch
from qvla_haq.offline_actions import file_sha256


class TrainingControlTests(unittest.TestCase):
    def test_schedule_endpoints_and_monotonic_segments(self):
        rates=[learning_rate_at(i,100,1e-6,1e-7,5,'warmup_cosine') for i in range(100)]
        self.assertAlmostEqual(rates[4],1e-6);self.assertAlmostEqual(rates[-1],1e-7)
        self.assertTrue(all(a<=b for a,b in zip(rates[:5],rates[1:5])))
        self.assertTrue(all(a>=b for a,b in zip(rates[5:],rates[6:])))
        self.assertEqual(learning_rate_at(1,2,1e-5,1e-7,0,'constant'),1e-5)

    def test_exact_optimizer_and_random_recovery(self):
        torch.manual_seed(29);random.seed(29);np.random.seed(29);rng=random.Random(29)
        model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        def update(model,optimizer,rng):
            x=torch.randn(4,3)*(rng.random()+random.random()+np.random.random())
            optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
        update(model,optimizer,rng)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt';identity={'steps':4,'source':'a'}
            save_checkpoint(path,model,optimizer,1,identity,rng,{'losses':[1]})
            update(model,optimizer,rng);expected={k:v.clone() for k,v in model.state_dict().items()}
            restored=torch.nn.Linear(3,2);other=torch.optim.AdamW(restored.parameters(),lr=.5);new_rng=random.Random(0)
            step,h=restore_checkpoint(path,restored,other,identity,new_rng)
            self.assertEqual(step,1);self.assertEqual(h['losses'],[1]);update(restored,other,new_rng)
            for k,v in expected.items():self.assertTrue(torch.equal(v,restored.state_dict()[k]),k)
            with self.assertRaisesRegex(ValueError,'settings changed'):restore_checkpoint(path,restored,other,{'source':'b'},new_rng)

    def test_stratified_unique_frames_cover_episodes_and_phases(self):
        rows=stratified_frames([(0,60,1,'task',0),(60,90,2,'task',0)],60,random.Random(29))
        self.assertEqual(len({(entry[2],rank) for entry,rank,phase in rows}),60)
        self.assertEqual({entry[2] for entry,_,_ in rows},{1,2})
        self.assertEqual([sum(phase==i for _,_,phase in rows) for i in range(3)],[20,20,20])
        self.assertTrue(all(rank+8<=entry[1] for entry,rank,_ in rows))
        with self.assertRaises(ValueError):stratified_frames([(0,9,1)],3,random.Random(0))

    def test_review_rejection_falls_back_to_gt_and_masks_teacher_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'actions.npy',np.ones((2,8,7),dtype=np.float32))
            rows=[dict(episode_index=3,frame_index=i,task_index=i,suite='libero_10',valid_length=8) for i in range(2)]
            manifest=dict(source_kind='openvla_oft_real_inference',teacher_execution_verified=True,
                action_space='libero_simulator_7d',fps=10,partition_sha256='p',split_sha256='s',rows=rows,actions_sha256=file_sha256(root/'actions.npy'))
            (root/'manifest.json').write_text(json.dumps(manifest))
            mask=np.zeros((2,8),dtype=bool);mask[1,:3]=True
            review=dict(teacher_actions_sha256=manifest['actions_sha256'],rows=rows,accepted_timestep_mask=mask.tolist())
            (root/'review.json').write_text(json.dumps(review))
            c=TeacherCache(root,partition_sha256='p',split_sha256='s',allowed_episodes=[3],review=root/'review.json')
            self.assertIsNone(c.choose(0,random.Random(0)));self.assertEqual(c.choose(1,random.Random(0)),1)
            b=teacher_batch({'action':torch.zeros(50,7)},lambda b:b,c.actions[1],8,c.accepted_masks[1])
            self.assertEqual(int((~b['action_is_pad']).sum()),3)
            self.assertTrue(torch.all(b['action'][0,3:]==0))

if __name__=='__main__':unittest.main()
