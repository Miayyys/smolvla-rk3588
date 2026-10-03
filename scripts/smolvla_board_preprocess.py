"""Pinned SmolVLA image, text and state preprocessing without Torch."""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


def resize_with_pad(image,height,width):
    """CHW float32; PyTorch bilinear align_corners=False, top/left padding."""
    channels,old_h,old_w=image.shape
    if (old_h,old_w)==(height,width):return image.copy()
    ratio=max(old_w/width,old_h/height)
    new_h,new_w=int(old_h/ratio),int(old_w/ratio)
    if min(new_h,new_w)<1:raise ValueError('Image aspect ratio is too extreme')
    def axis(old,new):
        coordinates=np.maximum((np.arange(new,dtype=np.float32)+np.float32(.5))*np.float32(old/new)-np.float32(.5),0)
        lower=np.floor(coordinates).astype(np.int64)
        return lower,np.minimum(lower+1,old-1),coordinates-lower.astype(np.float32)
    y0,y1,dy=axis(old_h,new_h);x0,x1,dx=axis(old_w,new_w)
    # Same weighted interpolation order as the Torch CUDA bilinear kernel.
    top=image[:,y0[:,None],x0[None,:]]*(1-dx)[None,None,:]+image[:,y0[:,None],x1[None,:]]*dx[None,None,:]
    bottom=image[:,y1[:,None],x0[None,:]]*(1-dx)[None,None,:]+image[:,y1[:,None],x1[None,:]]*dx[None,None,:]
    resized=top*(1-dy)[None,:,None]+bottom*dy[None,:,None]
    out=np.zeros((channels,height,width),dtype=np.float32)
    out[:,height-new_h:,width-new_w:]=resized
    return out


class SmolVLAPreprocessor:
    def __init__(self,tokenizer_json,tokenizer_config,processor_config,model_config,state_stats):
        self.model=json.loads(Path(model_config).read_text())
        processor=json.loads(Path(processor_config).read_text())
        steps={row['registry_name']:row['config'] for row in processor['steps']}
        text=steps['tokenizer_processor'];norm=steps['normalizer_processor']
        if self.model['adapt_to_pi_aloha'] or self.model['empty_cameras']!=0:
            raise ValueError('This adapter supports the pinned LIBERO configuration only')
        if norm['norm_map']['VISUAL']!='IDENTITY' or norm['norm_map']['STATE']!='MEAN_STD':
            raise ValueError('Unsupported normalization mode')
        self.eps=np.float32(norm['eps']);self.max_state_dim=self.model['max_state_dim']
        self.width,self.height=self.model['resize_imgs_with_padding']
        self.tokenizer=Tokenizer.from_file(str(tokenizer_json))
        cfg=json.loads(Path(tokenizer_config).read_text());pad=cfg['pad_token']
        if not isinstance(pad,str):pad=pad['content']
        pad_id=self.tokenizer.token_to_id(pad)
        if pad_id is None or text['padding']!='max_length' or text['padding_side']!='right' or not text['truncation']:
            raise ValueError('Unsupported tokenizer configuration')
        self.tokenizer.enable_truncation(max_length=text['max_length'],direction=cfg.get('truncation_side','right'))
        self.tokenizer.enable_padding(direction='right',pad_id=pad_id,pad_token=pad,length=text['max_length'])
        with np.load(state_stats,allow_pickle=False) as z:
            self.state_mean=z['mean'].copy();self.state_std=z['std'].copy()

    def tokenize(self,tasks):
        tasks=[str(t) if str(t).endswith('\n') else str(t)+'\n' for t in tasks]
        values=self.tokenizer.encode_batch(tasks,add_special_tokens=True)
        return np.array([v.ids for v in values],dtype=np.int64),np.array([v.attention_mask for v in values],dtype=bool)

    def __call__(self,image1,image2,state,task,noise):
        images=[]
        for raw in (image1,image2):
            raw=np.asarray(raw)
            if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[0]!=3:
                raise ValueError('Each camera must be RGB uint8 CHW')
            resized=resize_with_pad(raw.astype(np.float32)/np.float32(255),self.height,self.width)
            images.append((resized*np.float32(2)-np.float32(1))[None,:,:,:])
        state=np.asarray(state,dtype=np.float32)
        if state.shape!=self.state_mean.shape:raise ValueError('State size differs from checkpoint statistics')
        normalized=(state-self.state_mean)/(self.state_std+self.eps)
        padded=np.zeros((1,self.max_state_dim),dtype=np.float32);padded[0,:normalized.size]=normalized
        tokens,mask=self.tokenize([task])
        noise=np.asarray(noise,dtype=np.float32)
        if noise.shape!=(1,self.model['chunk_size'],self.model['max_action_dim']):raise ValueError('Noise shape mismatch')
        return {'images':np.stack(images),'image_masks':np.ones((2,1),dtype=bool),
                'lang_tokens':tokens,'lang_masks':mask,'state':padded,'noise':noise.copy()}
