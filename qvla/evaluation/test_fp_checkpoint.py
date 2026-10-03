
# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import unittest
import torch
from qvla.haq.fp_checkpoint import reconstruct_fp_state


class OverlayTests(unittest.TestCase):
    def test_exact_overlay_reconstruction_and_reject_missing_weights(self):
        base={'x.master_weight':torch.ones(2),'frozen':torch.tensor([3.])}
        saved={'x.weight':torch.tensor([2.,4.])}
        report={'master_storage_format':'trainable_fp_overlay',
                'training_scope':{'frozen_parameters_unchanged':True,'trainable_names':['x.master_weight']}}
        state=reconstruct_fp_state(saved,base,{'x'},report)
        self.assertTrue(torch.equal(state['x.master_weight'],saved['x.weight']))
        self.assertIs(state['frozen'],base['frozen'])
        with self.assertRaisesRegex(ValueError,'exactly cover'):
            reconstruct_fp_state({},base,{'x'},report)
        report['training_scope']['frozen_parameters_unchanged']=False
        with self.assertRaisesRegex(ValueError,'modified'):
            reconstruct_fp_state(saved,base,{'x'},report)


if __name__=='__main__':unittest.main()
