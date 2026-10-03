"""NumPy glue for the pinned SmolVLA cross-attention deployment graph."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import numpy as np


def bf16_round(x):
    x=np.ascontiguousarray(x,dtype=np.float32)
    bits=x.view(np.uint32)
    rounded=(bits+np.uint32(0x7fff)+((bits>>16)&1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def assemble_prefix(features,inputs,weights):
    width=features[0].shape[-1]
    parts=[x.astype(np.float32)*np.float32(width**0.5) for x in features]
    masks=[np.repeat(m[:,None],x.shape[1],axis=1) for m,x in zip(inputs['image_masks'],features)]
    if 'token_stored_weight' in weights:
        ids=inputs['lang_tokens']
        token=bf16_round(weights['token_stored_weight'][ids].astype(np.float32)*weights['token_weight_scale'][ids,None])
    else:
        token=weights['token_embedding'][inputs['lang_tokens']]
    language=bf16_round(token*np.float32(width**0.5))
    if 'state_stored_weight' in weights:
        scale=weights['state_activation_scale'];zero=weights['state_activation_zero']
        q=np.clip(np.rint(inputs['state'].astype(np.float32)/scale)+zero,-128,127).astype(np.int32)
        w=weights['state_stored_weight'].astype(np.int32)
        acc=q @ w.T-zero*w.sum(axis=1,dtype=np.int32)
        state=acc.astype(np.float32)*(scale*weights['state_weight_scale'])+weights['state_bias']
    else:
        state=inputs['state'] @ weights['state_weight'].T + weights['state_bias']
    parts.extend((language,state[:,None,:]));masks.extend((inputs['lang_masks'],np.ones(state.shape[:1]+(1,),dtype=bool)))
    prefix=np.ascontiguousarray(np.concatenate(parts,axis=1),dtype=np.float32)
    pad=np.concatenate(masks,axis=1).astype(bool)
    ar=np.zeros_like(pad,dtype=np.int64);ar[:,-1]=1
    groups=np.cumsum(ar,axis=1)
    attention=(groups[:,None,:]<=groups[:,:,None]) & pad[:,None,:] & pad[:,:,None]
    positions=np.cumsum(pad.astype(np.int64),axis=1)-1
    return prefix,pad,np.ascontiguousarray(attention),np.ascontiguousarray(positions)


def time_embedding(t,dim,min_period,max_period):
    fraction=np.linspace(0.,1.,dim//2,dtype=np.float64)
    periods=min_period*(max_period/min_period)**fraction
    angle=(1./periods*2*np.pi)[None,:]*np.asarray(t,dtype=np.float32).reshape(-1,1)
    return np.concatenate((np.sin(angle),np.cos(angle)),axis=1).astype(np.float32)


def postprocess(actions,weights):
    actions=actions[:,:,:weights['action_mean'].size]
    return actions*(weights['action_std']+np.float32(1e-8))+weights['action_mean']
