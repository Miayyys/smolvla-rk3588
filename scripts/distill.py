#!/usr/bin/env python3
"""QVLA distill command entry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvla.cli import main
if __name__ == '__main__':
    main('distill')
