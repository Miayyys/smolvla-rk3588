#!/usr/bin/env python3
"""Compare board preprocessing of 40 raw observations against original processor outputs."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from smolvla_board_preprocess import SmolVLAPreprocessor


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def metrics(actual,ref):
    d=actual.astype(np.float64)-ref.astype(np.float64)
    return {'mae':float(np.abs(d).mean()),'max_abs_error':float(np.abs(d).max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('assets','panel','reference','vision-panel','manifest','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();pm=json.loads(args.manifest.read_text());assets=args.assets
    for n,h in pm['config_hashes'].items():
        if digest(assets/n)!=h:raise ValueError('Asset hash mismatch: '+n)
    for path,name in [(args.panel,'raw_panel.npz'),(args.reference,'preprocess_reference.npz'),(assets/'state_stats.npz','state_stats.npz')]:
        if digest(path)!=pm['hashes'][name]:raise ValueError('Input/reference hash mismatch: '+name)
    if digest(args.vision_panel)!=pm['vision_panel_sha256']:raise ValueError('Vision reference hash mismatch')
    processor=SmolVLAPreprocessor(assets/'tokenizer.json',assets/'tokenizer_config.json',assets/'policy_preprocessor.json',assets/'config.json',assets/'state_stats.npz')
    with np.load(args.panel,allow_pickle=False) as z:raw={n:z[n].copy() for n in z.files}
    with np.load(args.reference,allow_pickle=False) as z:ref={n:z[n].copy() for n in z.files}
    with np.load(args.vision_panel,allow_pickle=False) as z:images=z['pixels'].copy()
    rows=[]
    for i in range(raw['state'].shape[0]):
        start=time.perf_counter();out=processor(raw['image1'][i],raw['image2'][i],raw['state'][i],str(raw['task'][i]),np.zeros((1,50,32),dtype=np.float32))
        elapsed=(time.perf_counter()-start)*1000
        np.testing.assert_array_equal(out['lang_tokens'][0],ref['lang_tokens'][i])
        np.testing.assert_array_equal(out['lang_masks'][0],ref['lang_masks'][i])
        np.testing.assert_allclose(out['state'][0],ref['state'][i],rtol=0,atol=1e-6)
        np.testing.assert_allclose(out['images'][:,0],images[2*i:2*i+2],rtol=0,atol=5e-7)
        rows.append({'index':i,'milliseconds':elapsed,'image_error':metrics(out['images'][:,0],images[2*i:2*i+2]),
                     'state_error':metrics(out['state'][0],ref['state'][i]),'tokens_and_masks_exact':True})
    tokens,masks=processor.tokenize(pm['extra_text_cases'])
    np.testing.assert_array_equal(tokens,ref['extra_token_ids']);np.testing.assert_array_equal(masks,ref['extra_token_masks'])
    report={'scope':'40 actual board raw-observation preprocess outputs vs original GPU processor; no closed-loop test',
        'status':'passed','rows':rows,'tokenizers':__import__('tokenizers').__version__,'numpy':np.__version__,
        'max_image_error':max(r['image_error']['max_abs_error'] for r in rows),
        'max_state_error':max(r['state_error']['max_abs_error'] for r in rows),
        'extra_text_ids_masks_exact':True,'preprocess_p50_ms':float(np.percentile([r['milliseconds'] for r in rows],50)),
        'preprocess_p95_ms':float(np.percentile([r['milliseconds'] for r in rows],95)),
        'manifest_sha256':digest(args.manifest),'raw_panel_sha256':digest(args.panel)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
