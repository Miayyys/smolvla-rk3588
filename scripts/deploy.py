#!/usr/bin/env python3
"""Verify, stage and run the final RK3588 release."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qvla.runtime.deployment import main
if __name__=='__main__':main()
