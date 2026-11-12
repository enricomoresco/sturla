#!/usr/bin/env python
"""Post-processing: analisi e grafici dai dati puliti.  Uso: python postprocessing.py -o out"""
import sys

from adcp_pd0.postprocess import main

if __name__ == "__main__":
    sys.exit(main())
