"""Build a deterministic tiny Llama fixture, never a SmolVLA substitute."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

root=Path(__file__).resolve().parents[1]
out=root/'runs/rkllm_dump_probe_v1'
out.mkdir(exist_ok=True)
model_dir=out/'hf_model'
model_dir.mkdir(exist_ok=True)
assets=root/'artifacts/transfer/smolvlm2_assets'
vocab=json.loads((assets/'tokenizer.json').read_text())
size=max([*vocab['model']['vocab'].values(), *[x['id'] for x in vocab['added_tokens']]])+1
torch.manual_seed(20261003)
torch.set_num_threads(2)
cfg=LlamaConfig(vocab_size=size,hidden_size=256,intermediate_size=512,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                max_position_embeddings=128,bos_token_id=1,eos_token_id=2,
                pad_token_id=0,tie_word_embeddings=True,rms_norm_eps=1e-6)
cfg._attn_implementation='eager'
model=LlamaForCausalLM(cfg).eval()
model.save_pretrained(model_dir)
for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json',
             'added_tokens.json','vocab.json','merges.txt'):
    if (assets/name).exists(): shutil.copy2(assets/name,model_dir/name)
torch.manual_seed(41)
embeds=torch.randn(1,8,256)*0.02
with torch.inference_mode():
    result=model(inputs_embeds=embeds,use_cache=True)
arrays={}
for i,layer in enumerate(result.past_key_values.layers):
    arrays[f'key_{i}']=layer.keys.numpy()
    arrays[f'value_{i}']=layer.values.numpy()
np.savez_compressed(out/'fp_kv_reference.npz',**arrays)
embeds.numpy().astype('<f4').tofile(out/'input_embeds.bin')
(out/'dataset.json').write_text(json.dumps([{'input':'Hello robot. Move the bowl.','target':''}]))
(out/'fixture.json').write_text(json.dumps({'scope':'random tiny causal Llama dump mechanism probe, not SmolVLA deployment',
    'seed':20261003,'input_seed':41,'layers':2,'hidden_size':256,'kv_heads':2,
    'head_dim':64,'tokens':8,'vocab_size':size,'reference_shapes':{k:list(v.shape) for k,v in arrays.items()}},indent=2)+'\n')
print(out,flush=True)
