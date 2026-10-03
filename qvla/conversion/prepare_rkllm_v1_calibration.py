"""Convert isolated V1 prefix calibration into official RKLLM embed samples."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);(root/'samples').mkdir(exist_ok=True)
source=a.source.resolve();rows=(source/'prefix_dataset.txt').read_text().splitlines()
episodes=json.loads((source/'export_report.json').read_text())['calibration_episode_ids']
assert len(rows)==len(episodes)==40
info=[];audit=[]
for i,row in enumerate(rows):
 paths=row.split();assert len(paths)==3
 prefix,mask,pos=[np.load(x,allow_pickle=False) for x in paths]
 valid=np.flatnonzero(mask[0].any(axis=0));assert valid[-1]==176
 x=np.ascontiguousarray(prefix[:,valid],dtype='f4');n=x.shape[1]
 m=mask[:,valid][:,:,valid]
 additive=np.where(m[:,None],np.float32(0),np.finfo(np.float32).min)
 inputs={'inputs_embeds':torch.from_numpy(x),'attention_mask':torch.from_numpy(additive),'position_ids':torch.from_numpy(pos[:,valid].copy())}
 assert np.array_equal(pos[:,valid],np.arange(n)[None])
 path=root/'samples'/f'{i:04d}.pkl'
 with path.open('wb') as f:pickle.dump(inputs,f)
 info.append({'sample':'samples/'+path.name,'token_nums':n})
 audit.append({'episode':episodes[i],'tokens':n,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'dataset.json').write_text(json.dumps(info,indent=2)+'\n')
(root/'calibration_report.json').write_text(json.dumps({'scope':'isolated training calibration only, compact original block mask, state last; official sample/token_nums pickle format','source_dataset_sha256':hashlib.sha256((source/'prefix_dataset.txt').read_bytes()).hexdigest(),'samples':audit},indent=2)+'\n')
