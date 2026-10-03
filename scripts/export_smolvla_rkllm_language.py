"""Extract the selected real language weights; mask compatibility remains a gate."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if digest != '4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81':
        raise ValueError('Expected selected V1 no-teacher QAT master')
    state = load_file(str(a.checkpoint))
    prefix = 'model.vlm_with_expert.vlm.model.text_model.'
    weights = {'model.' + k[len(prefix):]: v.float().contiguous()
               for k, v in state.items() if k.startswith(prefix)}
    del state
    layers = sorted({int(k.split('.')[2]) for k in weights if k.startswith('model.layers.')})
    assert layers == list(range(16)), layers
    # SmolVLA forward calls apply_rope(max_wavelength=10000), overriding asset config 100000.
    cfg = LlamaConfig(vocab_size=49280, hidden_size=960, intermediate_size=2560,
                      num_hidden_layers=16, num_attention_heads=15, num_key_value_heads=5,
                      head_dim=64, rms_norm_eps=1e-5, rope_theta=10000,
                      max_position_embeddings=8192, tie_word_embeddings=True,
                      pad_token_id=2, bos_token_id=1, eos_token_id=2)
    cfg.architectures = ['LlamaForCausalLM']
    # Logits are unused by VLA. Reuse embedding only to satisfy the causal-LM container.
    with torch.device('meta'):
        model = LlamaForCausalLM(cfg)
    expected = set(model.state_dict()) - {'lm_head.weight'}
    assert set(weights) == expected, (set(weights) - expected, expected - set(weights))
    a.output.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(a.output)
    save_file(weights, str(a.output / 'model.safetensors'), metadata={'format': 'pt'})
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'added_tokens.json', 'vocab.json', 'merges.txt'):
        if (a.assets / name).exists():
            shutil.copy2(a.assets / name, a.output / name)
    report = dict(checkpoint_sha256=digest, layers=16, tensor_count=len(weights),
                  rope_theta=10000, source_asset_rope_theta=100000,
                  mask_equivalence=False, scope='real weights in causal LM container; not equivalent deployment',
                  unused_lm_head='tied embedding; VLA does not consume logits')
    (a.output / 'extraction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
