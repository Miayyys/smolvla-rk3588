"""Run each compiled cost case sequentially, three rounds, core0, fixed protocol."""
import json,subprocess,sys,time
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent,
                    help='Directory containing cases.json and one folder per case')
parser.add_argument('--runner',type=Path,default=None,
                    help='RKNN Lite2 smoke runner; defaults to <root>/rknn_board_subgraph_smoke.py')
parser.add_argument('--python',default=sys.executable,
                    help='Python executable with rknnlite and numpy available')
parser.add_argument('--warmup',type=int,default=20)
parser.add_argument('--repeats',type=int,default=100)
parser.add_argument('--timeout',type=int,default=180)
parser.add_argument('--case-id',action='append',default=None,
                    help='Run only this case ID; may be repeated')
parser.add_argument('--force',action='store_true',
                    help='Overwrite existing round reports for selected cases')
args=parser.parse_args()
root=args.root.resolve()
runner=(args.runner or root/'rknn_board_subgraph_smoke.py').resolve()
cases=json.loads((root/'cases.json').read_text())
if args.case_id is not None:
 cases=[case for case in cases if case['case_id'] in set(args.case_id)]
 if not cases: parser.error('No cases matched --case-id')
for rnd in range(3):
 for case in (cases if rnd%2==0 else list(reversed(cases))):
  d=root/case['case_id']
  if not (d/'model.rknn').exists():continue
  result=d/f'board_{rnd}.json'
  if result.exists() and not args.force:continue
  thermal={str(p):p.read_text().strip() for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
  begin=time.monotonic()
  command=[args.python,str(runner),'--model',str(d/'model.rknn'),'--input',str(d/'input.npy'),
           '--output',str(result),'--warmup',str(args.warmup),'--repeats',str(args.repeats)]
  if (d/'reference.npy').exists():command.extend(['--reference',str(d/'reference.npy')])
  with (d/f'board_{rnd}.log').open('w') as log:
   try:ret=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout).returncode
   except subprocess.TimeoutExpired:ret=124
  if not result.exists():result.write_text(json.dumps({'status':'failed','returncode':ret}))
  r=json.loads(result.read_text());r.update(thermal_before=thermal,process_elapsed_seconds=time.monotonic()-begin,process_returncode=ret);result.write_text(json.dumps(r,indent=2))
  print(rnd,case['case_id'],r['status'],flush=True)
