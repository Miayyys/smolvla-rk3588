"""Measure padding compaction and mask alternatives on the selected real language module."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import json, numpy as np, torch
from pathlib import Path
from transformers import LlamaForCausalLM
torch.set_num_threads(4)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--language-root',type=Path,default=Path('runs/selected_language_rkllm_v1'))
p.add_argument('--boundary-root',type=Path,default=Path('runs/qat_v1_rknn_deploy_v1'))
p.add_argument('--output',type=Path,default=Path('runs/rkllm_mask_resolution_v1'))
a=p.parse_args();r=a.language_root;o=a.output;o.mkdir(parents=True,exist_ok=True)
b=np.load(a.boundary_root/'prefix_boundary.npz');ref=np.load(r/'original_mask_reference.npz')
m=b['attention_mask'][0];keep=m.any(axis=0);cm=m[keep][:,keep];pos=b['position_ids'][:,keep];x=torch.from_numpy(b['prefix'][:,keep]).float()
assert np.array_equal(pos,np.arange(keep.sum())[None])
model=LlamaForCausalLM.from_pretrained(r/'hf_model',dtype=torch.float32,attn_implementation='eager').eval()
rows={}
for mode,mask in [('compact_original',cm),('compact_causal',np.tril(np.ones_like(cm))),('compact_bidirectional',np.ones_like(cm))]:
 mask=torch.where(torch.from_numpy(mask)[None,None],0.,torch.finfo(torch.float32).min)
 with torch.inference_mode():y=model.model(inputs_embeds=x,attention_mask=mask,position_ids=torch.from_numpy(pos),use_cache=True)
 metrics=[]
 for i,l in enumerate(y.past_key_values.layers):
  for name,v in [('key',l.keys),('value',l.values)]:
   d=v.numpy().astype('f8')-ref[f'{name}_{i}'][:,:,keep,:].astype('f8')
   metrics.append({'name':f'{name}_{i}','mae':float(np.abs(d).mean()),'max_abs':float(np.abs(d).max())})
 rows[mode]={'outputs':metrics,'mean_mae':float(np.mean([z['mae'] for z in metrics])),'max_abs':max(z['max_abs'] for z in metrics)}
report={'scope':'one real selected language observation, FP32 semantic diagnostic only','tokens_before':len(keep),'tokens_after':int(keep.sum()),'positions_contiguous':True,'prefix_bidirectional':bool(cm[:-1,:-1].all()),'prefix_sees_state':bool(cm[:-1,-1].any()),'state_sees_all':bool(cm[-1].all()),'comparisons':rows}
(o/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in report.items() if k!='comparisons'},'comparisons':{k:{a:b for a,b in v.items() if a!='outputs'} for k,v in rows.items()}}))
