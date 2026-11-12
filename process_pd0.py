#!/usr/bin/env python
"""Strumenti per dati ADCP Sentinel V (.pd0).

  python process_pd0.py inspect FILE.pd0 [--range 45%:55%]   ispezione veloce, non scrive niente
  python preprocessing.py FILE.pd0 -o out                    dati puliti in NetCDF (L2)
  python postprocessing.py -o out                            analisi, grafici, risultati (L3)

`process_pd0.py preprocess ...` e `process_pd0.py postprocess ...` equivalgono ai due script.
"""
import argparse
import sys

from adcp_pd0 import inspect_files


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "preprocess":
        from adcp_pd0.preprocess import main as m
        return m(argv[1:])
    if argv and argv[0] == "postprocess":
        from adcp_pd0.postprocess import main as m
        return m(argv[1:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="ispezione veloce dei file (nessun output scritto)")
    pi.add_argument("inputs", nargs="+")
    pi.add_argument("--max-mb", type=float, default=None, help="analizza solo N MB (dall'inizio della porzione)")
    pi.add_argument("--range", dest="byte_range", default=None,
                    help="porzione di ogni file, INIZIO:FINE in %% o MB (es. 45%%:55%%)")
    sub.add_parser("preprocess", help="vedi preprocessing.py -h", add_help=False)
    sub.add_parser("postprocess", help="vedi postprocessing.py -h", add_help=False)
    a = p.parse_args(argv)
    inspect_files(a.inputs, a.max_mb, a.byte_range)
    return 0


if __name__ == "__main__":
    sys.exit(main())
