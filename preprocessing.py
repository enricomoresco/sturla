#!/usr/bin/env python
"""Pre-processing: dal file PD0 ai dati puliti in NetCDF.  Uso: python preprocessing.py file.pd0 -o out"""
import sys

from adcp_pd0.preprocess import main

if __name__ == "__main__":
    sys.exit(main())
