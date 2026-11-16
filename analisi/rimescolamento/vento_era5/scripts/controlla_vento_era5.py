#!/usr/bin/env python3
"""Controlli prima delle run con il vento ERA5. Solo libreria standard.

  python3 scripts/controlla_vento_era5.py <cartella Sturla_GOTM_v7_2026> CASO [CASO ...]

Per ogni caso controlla che: tutti i file citati nel gotm.yaml esistano; la convezione CVMix
sia accesa; i casi *_ERA5 leggano sforzo e vento da data/gotm_era5 e i casi *_CMEMS dai file
CMEMS di sempre; i file ERA5 coprano il periodo della run.
"""
import re
import sys
from datetime import datetime
from pathlib import Path


class CheckError(Exception):
    pass


def tempi(p):
    t = []
    for riga in p.read_text().splitlines():
        s = riga.split()
        if len(s) >= 4 and re.match(r"^\d{4}-\d\d-\d\d$", s[0]):
            t.append(datetime.strptime(s[0] + " " + s[1], "%Y-%m-%d %H:%M:%S"))
            [float(x) for x in s[2:4]]
    if not t or t != sorted(t) or len(t) != len(set(t)):
        raise CheckError(f"{p}: date mancanti, duplicate o fuori ordine")
    return t


def run(root, casi):
    era5 = {k: root / "data" / "gotm_era5" / f for k, f in
            (("tau", "momentum_flux_era5.dat"), ("vento", "wind10_era5.dat"))}
    copertura = {k: tempi(p) for k, p in era5.items()}
    print(f"OK file ERA5: {copertura['tau'][0]} -> {copertura['tau'][-1]}")
    for caso in casi:
        y = root / "templates" / caso / "gotm.yaml"
        if not y.is_file():
            raise CheckError(f"manca {y}")
        txt = y.read_text()
        if "    convection:\n      use: true" not in txt:
            raise CheckError(f"{caso}: convezione CVMix non accesa")
        for f in re.findall(r"^\s*file:\s*(\S+)", txt, re.M):
            if not (y.parent / f).is_file():
                raise CheckError(f"{caso}: file non trovato {f}")
        tau = re.findall(r"^\s*file:\s*(\S*momentum_flux\S*)", txt, re.M)
        u10 = re.findall(r"^\s*file:\s*(\S*wind10\S*)", txt, re.M)
        if len(tau) != 2 or len(u10) != 2:
            raise CheckError(f"{caso}: attese 2 righe per tx/ty e 2 per u10/v10")
        vuole_era5 = caso.endswith("_ERA5")
        if any(("gotm_era5" in f) != vuole_era5 for f in tau + u10):
            raise CheckError(f"{caso}: sorgente del vento non coerente con il nome del caso")
        t0 = datetime.strptime(re.search(r"^\s*start:\s*(.+)$", txt, re.M).group(1).strip(), "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(re.search(r"^\s*stop:\s*(.+)$", txt, re.M).group(1).strip(), "%Y-%m-%d %H:%M:%S")
        if vuole_era5:
            for k, t in copertura.items():
                if t[0] > t0 or t[-1] < t1:
                    raise CheckError(f"{caso}: {era5[k].name} non copre {t0} -> {t1}")
        print(f"OK {caso:20s} {t0} -> {t1}  vento {'ERA5' if vuole_era5 else 'CMEMS'}")
    print("PRECHECK OK")


if __name__ == "__main__":
    try:
        run(Path(sys.argv[1]).resolve(), sys.argv[2:])
    except (CheckError, ValueError, IndexError, OSError, AttributeError) as e:
        print("ERRORE PRECHECK:", e, file=sys.stderr)
        sys.exit(1)
