"""Persistent SmolVLA runtime for pinned FP16 or mixed RKNN deployments."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import json
import time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
from qvla.runtime.smolvla_board_preprocess import SmolVLAPreprocessor
from qvla.runtime.smolvla_numpy_glue import assemble_prefix, time_embedding, postprocess
from qvla.runtime.rknn_board_full_replay import digest


class BoardSmolVLA:
    def __init__(self,root):
        root=Path(root);self.root=root;self.config=json.loads((root/'replay.json').read_text())
        pm=json.loads((root/'preprocess_export.json').read_text())
        for name,h in pm['config_hashes'].items():
            if digest(root/name)!=h:raise ValueError('Preprocessing asset changed: '+name)
        if digest(root/'state_stats.npz')!=pm['hashes']['state_stats.npz']:raise ValueError('State statistics changed')
        if digest(root/'cpu_weights.npz')!=self.config['hashes']['cpu_weights.npz']:raise ValueError('CPU weights changed')
        self.pre=SmolVLAPreprocessor(root/'tokenizer.json',root/'tokenizer_config.json',root/'policy_preprocessor.json',root/'config.json',root/'state_stats.npz')
        with np.load(root/'cpu_weights.npz') as z:self.weights={n:z[n].copy() for n in z.files}
        manifest=root/'deployment_manifest.json';self.prefix_scale=None;self.language=None
        if manifest.exists():
            dm=json.loads(manifest.read_text())
            if len(dm['graphs'])!=3:raise ValueError('Expected vision, prefix and expert graphs')
            names=[g['filename'] for g in dm['graphs']]
            if any(Path(n).name!=n for n in names):raise ValueError('Invalid deployment graph filename')
            expected=[g['sha256'] for g in dm['graphs']]
            if dm['cpu_weights_sha256']!=digest(root/'cpu_weights.npz'):raise ValueError('Deployment CPU weights changed')
            if dm.get('prefix_input_scale'):
                asset=dm['prefix_input_scale'];name=asset['filename']
                if Path(name).name!=name or digest(root/name)!=asset['sha256']:raise ValueError('Prefix scale changed')
                self.prefix_scale=np.load(root/name,allow_pickle=False)
                if self.prefix_scale.shape!=(1,177,960) or not np.isfinite(self.prefix_scale).all() or np.any(self.prefix_scale<=0):
                    raise ValueError('Invalid prefix scale')
        else:
            names=['vision_connector_fp16.rknn','prefix_with_kv_fp16.rknn','expert_step_v2_fp16.rknn']
            expected=['f486c5de7f0bc085e9bb1157db761003b9d81d8c2e1e83cd11fb53942b209434',
                  '9812496c93b2125b7a3c72c5b69ec3c63ef6ccfeb9090f137e46c4004e80706c',
                  'c85c63773b2fea2b2fa17a82ecb1538c68f50a0e6f8fe032cccfb5ad074c31eb']
        language=dm.get('language_backend') if manifest.exists() else None
        if language:
            from qvla.runtime.smolvla_rkllm_backend import RKLLMLanguage
            self.language=RKLLMLanguage(root,language,digest)
        partitions=dm.get('partitioned_graphs',{}) if manifest.exists() else {}
        self.partition_models={}
        roles=['vision','prefix','expert']
        self.hashes={n:digest(root/n) for i,n in enumerate(names) if not (self.language and i==1) and roles[i] not in partitions};self.models=[]
        for index,(name,h) in enumerate(zip(names,expected)):
            if roles[index] in partitions:
                if index==0 or (index==1 and self.language):raise ValueError('Unsupported partition backend combination')
                from qvla.runtime.smolvla_rknn_partitions import RKNNPartitions
                part=RKNNPartitions(root,partitions[roles[index]],digest)
                self.partition_models[roles[index]]=part;self.hashes.update(part.hashes)
                self.models.append(None);continue
            if index==1 and self.language:
                self.models.append(None);continue
            if self.hashes[name]!=h:raise ValueError('RKNN graph changed: '+name)
            m=RKNNLite();self.models.append(m)
            if m.load_rknn(str(root/name)) or m.init_runtime(core_mask=RKNNLite.NPU_CORE_0):raise RuntimeError('RKNN init failed')

    def predict(self,raw):
        start=time.perf_counter()
        x=self.pre(raw['image1'],raw['image2'],raw['state'],str(raw['task'].item()),raw['noise'])
        pre_ms=(time.perf_counter()-start)*1000
        def infer(m,values):
            y=m.inference(inputs=[np.ascontiguousarray(a) for a in values],data_format=['nchw']*len(values))
            if y is None or any(not np.isfinite(a).all() for a in y):raise RuntimeError('Invalid RKNN output')
            return y
        features=[infer(self.models[0],[image])[0] for image in x['images']]
        prefix,pad,mask,pos=assemble_prefix(features,x,self.weights)
        prefix_input=prefix if self.prefix_scale is None else prefix/self.prefix_scale
        outputs=([None,*self.language.infer(prefix,pad)] if self.language else
                 self.partition_models['prefix'].infer([prefix_input,mask,pos]) if 'prefix' in self.partition_models else
                 infer(self.models[1],[prefix_input,mask,pos]))
        if len(outputs)!=33:raise ValueError('Expected all 16-layer KV tensors')
        actions=x['noise'].copy();dt=-1./self.config['num_steps']
        for step in range(self.config['num_steps']):
            embedding=time_embedding(np.array([1.+step*dt],dtype=np.float32),self.config['expert_hidden_size'],self.config['min_period'],self.config['max_period'])
            expert_inputs=[actions,embedding,pad,*outputs[1:]]
            velocity=(self.partition_models['expert'].infer(expert_inputs) if 'expert' in self.partition_models else infer(self.models[2],expert_inputs))[0]
            if velocity.shape!=actions.shape:raise ValueError('Velocity shape mismatch')
            actions=actions+np.float32(dt)*velocity
        final=postprocess(actions,self.weights)
        calls={'vision':2,
               'prefix':len(self.partition_models['prefix'].models) if 'prefix' in self.partition_models else 1,
               'expert':self.config['num_steps']*(len(self.partition_models['expert'].models) if 'expert' in self.partition_models else 1)}
        return final,{'inference_ms':(time.perf_counter()-start)*1000,'preprocess_ms':pre_ms,'language_backend':'rkllm' if self.language else 'rknn','calls':calls}

    def close(self):
        if self.language:self.language.close()
        for part in self.partition_models.values():part.close()
        for m in reversed(self.models):
            if m is not None:m.release()
