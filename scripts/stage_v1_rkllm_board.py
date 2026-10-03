"""Stage V1 RKNN vision/expert + adapted W8A8 RKLLM language, resumably."""
import argparse,copy,hashlib,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--language',type=Path,default=Path('runs/v1_rkllm_w8a8_g128_v1/language_w8a8.rkllm'))
p.add_argument('--prepare-only',action='store_true',help='Prepare small code/metadata; user uploads the large language model separately')
p.add_argument('--pool-bytes',type=int,default=32022528,help='W8A8 working-pool bytes measured from native allocation log')
a=p.parse_args();project=Path(__file__).resolve().parents[1];language=a.language.resolve()
report=json.loads(language.with_suffix('.compile.json').read_text())
if report.get('status')!='compiled' or report['quantized_dtype']!='w8a8':raise ValueError('Expected compiled W8A8 language')
sha=hashlib.sha256(language.read_bytes()).hexdigest()
if sha!=report['sha256']:raise ValueError('Language model changed')
extraction=json.loads((project/'runs/selected_language_rkllm_v1/hf_model/extraction.json').read_text())
m=json.loads((project/'runs/v1_exact_precision_v1/deployment_manifest.json').read_text())
if extraction['checkpoint_sha256']!=m['selected_master_sha256']:raise ValueError('Language and expert master mismatch')
old=json.loads((project/'runs/rkllm_native_patch_v1/allocation_fix/deployment_manifest.json').read_text())
backend=copy.deepcopy(old['language_backend']);old_name=backend['model'];backend['assets'].pop(old_name)
backend.update(model=language.name,validation_status='new_W8A8_backend_not_yet_board_validated')
backend['assets'][language.name]=sha
backend['allocation_control']['pool_bytes']=a.pool_bytes
m['partitioned_graphs'].pop('prefix');m['language_backend']=backend
m['graphs'][1]={'role':'prefix','filename':language.name,'sha256':sha,'bytes':language.stat().st_size}
m['scope']='V1 trained master, adapted language W8A8 RKLLM; RKNN V1 vision and expert formats retained'
m['deployment_override']={'language_linear_format':'RKLLM w8a8 normal optimization1','original_INT16_DFP_down3_replaced':True,'not_exact_original_V1_precision':True}
m['board_execution']='not_measured'
m['model_parameter_bytes']=m['graphs'][0]['bytes']+language.stat().st_size+sum(x['bytes'] for x in m['partitioned_graphs']['expert']['parts'])+(project/'runs/qat_v1_rknn_deploy_v1/cpu_weights.npz').stat().st_size
out=language.parent/'mixed_backend';out.mkdir(exist_ok=True)
(out/'deployment_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
root='/dev/shm/qvla_v1_rkllm_language_v1';base='/dev/shm/qvla_v1_exact_precision_v1';llm='/root/qvla_board_test/rkllm_native_patch_v1'
remote='''from pathlib import Path
import json
r=Path(__ROOT__);base=Path(__BASE__);llm=Path(__LLM__);r.mkdir(exist_ok=True)
pm=json.loads((base/'preprocess_export.json').read_text())
names=['cpu_weights.npz','state_stats.npz','raw_inputs.npz','fp_reference.npz','replay.json','preprocess_export.json','smolvla_numpy_glue.py','smolvla_board_preprocess.py','rknn_board_full_replay.py','vision_selected.rknn','expert_before_projection.rknn','expert_projection.rknn','expert_after_projection.rknn','rknn_api.h','librknn_bf16_projection.so',*pm['config_hashes']]
for name in dict.fromkeys(names):
 source=base/name
 if not source.is_file():raise FileNotFoundError(source)
 target=r/name
 if not target.exists():target.symlink_to(source)
 if target.resolve()!=source.resolve():raise ValueError('Unexpected existing asset '+name)
for name in ['rkllm_worker','libqvla_rknpu_alloc.so','patched_lib']:
 source=llm/name;target=r/name
 if not source.exists():raise FileNotFoundError(source)
 if not target.exists():target.symlink_to(source)
 if target.resolve()!=source.resolve():raise ValueError('Unexpected RKLLM asset '+name)
'''.replace('__ROOT__',repr(root)).replace('__BASE__',repr(base)).replace('__LLM__',repr(llm))
ssh=['ssh','-F','/dev/null','-o','ProxyCommand=none','-o','ConnectTimeout=8'];board='root@10.42.0.252'
subprocess.run([*ssh,board,'python3 -'],input=remote,text=True,check=True)
files=[out/'deployment_manifest.json',*[project/'scripts'/n for n in ['smolvla_board_runtime.py','smolvla_rknn_partitions.py','smolvla_bf16_projection.py','smolvla_rkllm_backend.py','serve_smolvla_board_stdio.py','verify_v1_partitioned_replay.py','verify_smolvla_rkllm_persistent.py']]]
if not a.prepare_only:files.append(language)
subprocess.run(['rsync','-a','--checksum','--partial','--info=progress2','-e',shlex.join(ssh),*[str(x) for x in files],board+':'+root+'/'],check=True)
print(json.dumps({'root':root,'model_parameter_bytes':m['model_parameter_bytes'],'compression_vs_original':1-m['model_parameter_bytes']/906712520,'language_uploaded':not a.prepare_only,'board_execution':'not_measured'}))
