#!/usr/bin/env python3
"""Check frozen scopes, teacher contracts and checkpoint/training controls."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    modules=['fp_checkpoint','training_scope','module_recovery','training_control','distillation_contract']
    suite=unittest.defaultTestLoader.loadTestsFromNames(['qvla.evaluation.test_'+name for name in modules])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
