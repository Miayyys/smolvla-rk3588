"""Describe all seven executed RKNN graphs without silently remapping V1 formats."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--compiled',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
a=p.parse_args();root=a.compiled;old=json.loads((a.baseline/'deployment_manifest.json').read_text())
candidate=json.loads(a.candidate.read_text())
manifest={'scope':'V1 requested module formats retained by RKNN partitions; native quantizer equivalence and board quality unverified',
 'selected_master_sha256':old['selected_master_sha256'],'selected_pack_sha256':old['selected_pack_sha256'],
 'cpu_weights_sha256':old['cpu_weights_sha256'],'assignment_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
 'graphs':[old['graphs'][0]],'partitioned_graphs':{},'board_execution':'not_measured',
 'source_local_pack_equivalence':False,'native_quantizer_calibration_and_fusion_differ':True,
 'compute_precision_audit':{'expert_v3':'FLOAT16 NPU Conv','expert_gate10':'BFLOAT16 NPU Conv',
 'prefix_down3':'INT16 NPU Conv with global w16a16i_dfp configuration'},'requested_assignment':candidate['assignment']}
rows=[dict(old['graphs'][0])]
for kind in ['prefix','expert']:
 conv=json.loads((root/(kind+'_conversion.json')).read_text());split=json.loads((root/(kind+'_split_report.json')).read_text())
 meta=json.loads((root/'source_metadata'/(kind+'_export.json')).read_text())
 if conv['status']!='compiled' or split['fp_partition_parity']['status']!='passed':raise ValueError('Compilation/parity not passed')
 group={'input_names':meta['input_names'],'output_names':meta['output_names'],'parts':[]}
 for part in ['before_projection','projection','after_projection']:
  r=conv['parts'][part];s=split['parts'][part]
  group['parts'].append({**r,'input_names':s['inputs'],'output_names':s['outputs'],
    'input_dtypes':s['input_dtypes'],'output_dtypes':s['output_dtypes'],
    'execution_backend':'native_bf16_projection' if kind=='expert' and part=='projection' else 'rknnlite'})
  rows.append(dict(r))
 manifest['partitioned_graphs'][kind]=group
 first=dict(conv['parts']['before_projection']);first['role']=kind;manifest['graphs'].append(first)
manifest['model_parameter_bytes']=sum(r['bytes'] for r in rows)+(a.baseline/'cpu_weights.npz').stat().st_size
(root/'deployment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'model_parameter_inventory.json').write_text(json.dumps({'scope':'seven graph files plus existing selected CPU weights; excludes config/runtime code and shared runtime libraries',
 'files':rows+[{'filename':'cpu_weights.npz','bytes':(a.baseline/'cpu_weights.npz').stat().st_size,'sha256':old['cpu_weights_sha256']}],
 'bytes':manifest['model_parameter_bytes']},indent=2)+'\n')
print(json.dumps({'model_parameter_bytes':manifest['model_parameter_bytes'],'new_graph_download_bytes':sum(r['bytes'] for r in rows[1:])}),flush=True)
