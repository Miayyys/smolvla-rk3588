"""Compare 40 saved initial observations with checkpoint processors and board NumPy preprocessing."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from qvla.runtime.smolvla_board_preprocess import SmolVLAPreprocessor
from qvla.runtime.smolvla_numpy_glue import postprocess
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/final_model_comparison_v1';OUT=RUN/'evaluation_audit'
torch.set_num_threads(2)
model=ROOT/'artifacts/transfer/model';assets=ROOT/'artifacts/transfer/smolvlm2_assets'
config=SmolVLAConfig.from_pretrained(model);config.device='cpu'
pre,post=make_pre_post_processors(config,str(model),preprocessor_overrides={'tokenizer_processor':{'tokenizer_name':str(assets.resolve())},'device_processor':{'device':'cpu'}})
board=SmolVLAPreprocessor(assets/'tokenizer.json',assets/'tokenizer_config.json',model/'policy_preprocessor.json',model/'config.json',ROOT/'runs/smolvla_raw_board_v1/state_stats.npz')
wrapper=SimpleNamespace(config=config);summary=json.loads((RUN/'quality40/summary.json').read_text());rows=[]
for pair in summary['pairs']:
 p=RUN/'quality40'/f"fp_{pair['suite']}_{pair['task_id']}"/'input_000.npz'
 with np.load(p) as z:raw={k:z[k].copy() for k in z.files}
 batch=pre({'observation.images.image':torch.from_numpy(raw['image1'][None]).float()/255,'observation.images.image2':torch.from_numpy(raw['image2'][None]).float()/255,'observation.state':torch.from_numpy(raw['state'][None]),'task':str(raw['task'].item())})
 images,masks=SmolVLAPolicy.prepare_images(wrapper,batch);state=SmolVLAPolicy.prepare_state(wrapper,batch)
 actual=board(raw['image1'],raw['image2'],raw['state'],str(raw['task'].item()),raw['noise'])
 expected_images=np.stack([x.numpy() for x in images]);expected_masks=np.stack([x.numpy() for x in masks])
 image_error=float(np.max(np.abs(expected_images-actual['images'])));state_error=float(np.max(np.abs(state.numpy()-actual['state'])))
 np.testing.assert_allclose(expected_images,actual['images'],rtol=0,atol=5e-7);np.testing.assert_allclose(state.numpy(),actual['state'],rtol=0,atol=1e-6)
 np.testing.assert_array_equal(expected_masks,actual['image_masks']);np.testing.assert_array_equal(batch['observation.language.tokens'].numpy(),actual['lang_tokens']);np.testing.assert_array_equal(batch['observation.language.attention_mask'].numpy(),actual['lang_masks'])
 rows.append(dict(suite=pair['suite'],task_id=pair['task_id'],image_max_abs=image_error,state_max_abs=state_error,text_ids_and_masks_exact=True))
with np.load(ROOT/'runs/qat_v1_rknn_deploy_v1/cpu_weights.npz') as z:weights={k:z[k].copy() for k in ['action_mean','action_std']}
x=np.random.default_rng(42).normal(size=(1,50,7)).astype('f4');a=post(torch.from_numpy(x)).numpy();b=postprocess(x,weights)
np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
report={'passed':True,'rows':rows,'scope':'checkpoint CPU preprocessing vs deployed NumPy formulas on 40 actual initial observations; no network execution',
 'max_image_error':max(x['image_max_abs'] for x in rows),'max_state_error':max(x['state_max_abs'] for x in rows),'postprocessor_max_abs':float(np.max(np.abs(a-b)))}
(OUT/'preprocessing_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
