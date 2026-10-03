"""Compare the mixed PTQ candidate with saved paired-protocol FP results."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp-root',type=Path,required=True)
    p.add_argument('--ptq-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[];suites={};hashes={}
    for suite in ('libero_spatial','libero_object','libero_goal','libero_10'):
        values={}
        for label,root in [('fp',a.fp_root),('ptq',a.ptq_root)]:
            path=root/suite/'eval_info.json';data=json.loads(path.read_text())
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            tasks=data['per_task']
            if data['overall']['n_episodes']!=10 or len(tasks)!=10:raise ValueError('Expected 10 episodes')
            values[label]={}
            for task in tasks:
                outcomes=task['metrics']['successes'];key=task['task_id']
                if len(outcomes)!=1 or task['task_group']!=suite or key in values[label]:raise ValueError('Invalid task result')
                values[label][key]=bool(outcomes[0])
            if set(values[label])!=set(range(10)):raise ValueError('Missing tasks')
        suites[suite]={label:sum(v.values()) for label,v in values.items()}
        rows.extend({'suite':suite,'task_id':i,'fp':values['fp'][i],'ptq':values['ptq'][i]} for i in range(10))
    report={'scope':'40 development tasks, one episode/task; historical FP under paired protocol, not contemporaneous rerun',
            'totals':{label:sum(r[label] for r in rows) for label in ('fp','ptq')},'suites':suites,
            'regressions':[r for r in rows if r['fp'] and not r['ptq']],
            'improvements':[r for r in rows if r['ptq'] and not r['fp']], 'tasks':rows,'input_hashes':hashes}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tasks','input_hashes')},indent=2))

if __name__=='__main__':main()
