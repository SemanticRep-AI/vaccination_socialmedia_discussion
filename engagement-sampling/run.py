#!/usr/bin/env python3
"""Convenience entry point: python run.py --config config/vaccine_2026.yaml"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from engagement_sampling.cli import main

if __name__ == "__main__":
    sys.exit(main())
