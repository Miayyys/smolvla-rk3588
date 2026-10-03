"""Sequential background board resource benchmarks then paired GPU/board 40-task evaluation."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
from qvla.evaluation.report_final_model_comparison import render
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/final_model_comparison_v1';RUN.mkdir(exist_ok=True)
SSH=['ssh','-F','/dev/null','-o','ProxyCommand=none','-o','ConnectTimeout=8'];BOARD='root@10.42.0.252'
BEGIN=time.monotonic()
def status(stage,state,**extra):
    value={'stage':stage,'state':state,'pid':os.getpid(),'elapsed_seconds':time.monotonic()-BEGIN,**extra}
    p=RUN/'progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p);render();print(json.dumps(value),flush=True)
def run(stage,command):
    status(stage,'running',command=command)
    with (RUN/(stage+'.log')).open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
        while proc.poll() is None:
            status(stage,'running',child_pid=proc.pid);time.sleep(10)
        if proc.returncode:raise RuntimeError(stage+' failed with exit '+str(proc.returncode))
    status(stage,'stage_completed')
def copy_reports(root,name):
    dest=RUN/name;dest.mkdir(exist_ok=True)
    subprocess.run(['scp','-O','-F','/dev/null','-o','ProxyCommand=none',*[BOARD+':'+root+'/benchmark/'+n for n in ['benchmark.json','resource_samples.jsonl','actions.npz']],str(dest)],check=True)
    render()
try:
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'qvla/evaluation/run_smolvla_board_libero.py',ROOT/'qvla/evaluation/benchmark_smolvla_board_resources.py',RUN/'final_deployment_manifest.json',ROOT/'artifacts/transfer/model/model.safetensors']}
    (RUN/'source_hashes.json').write_text(json.dumps(frozen,indent=2)+'\n')
    for name,root in [('original_board','/dev/shm/qvla_original_fp16_benchmark_v1'),('final_board','/dev/shm/qvla_v1_rkllm_original_fp16_vision_v1')]:
        command='cd '+shlex.quote(root)+' && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site python3 -u benchmark_smolvla_board_resources.py --root . --output benchmark --warmup 3 --repeats 20'
        run(name,SSH+[BOARD,command]);copy_reports(root,name)
    run('quality40',[sys.executable,'-u',str(ROOT/'qvla/evaluation/run_smolvla_board_libero.py'),
        '--board-root','/dev/shm/qvla_v1_rkllm_original_fp16_vision_v1','--suites','libero_spatial','libero_object','libero_goal','libero_10',
        '--task-ids',*[str(i) for i in range(10)],'--seed','0',
        '--replay-inputs',str(ROOT/'runs/smolvla_raw_board_v1/raw_inputs.npz'),
        '--replay-reference',str(ROOT/'runs/v1_rkllm_original_fp16_vision_v1/raw_replay_reference.npz'),
        '--output',str(RUN/'quality40')])
    status('pipeline','completed')
except Exception as e:
    status('pipeline','failed',error=str(e));raise
