#!/usr/bin/env python3
"""Serve complete board inference through an existing SSH stdio stream."""
import argparse
import base64
import io
import json
import resource
import sys
from pathlib import Path
import numpy as np
from smolvla_board_runtime import BoardSmolVLA


def reply(value):print('QVLA_REPLY '+json.dumps(value),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    runtime=BoardSmolVLA(args.root)
    try:
        reply({'ready':True,'checkpoint_sha256':runtime.config['checkpoint_sha256'],'graphs':runtime.hashes,'core':'NPU_CORE_0','language_backend':'rkllm' if runtime.language else 'rknn'})
        for line in sys.stdin:
            req=json.loads(line)
            if req.get('quit'):break
            try:
                with np.load(io.BytesIO(base64.b64decode(req['npz'])),allow_pickle=False) as z:raw={n:z[n].copy() for n in z.files}
                actions,timing=runtime.predict(raw)
                reply({'id':req['id'],'actions':actions.tolist(),**timing,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            except Exception as e:
                reply({'id':req.get('id'),'error':repr(e)});raise
    finally:runtime.close()


if __name__=='__main__':main()
