"""Prepara le run GOTM con il vento ERA5 al posto del vento CMEMS (8 casi; slide 49 e 51 della
presentazione), partendo dalle run gia' fatte.

  python analisi/rimescolamento/gotm_vento_era5.py --dati ... --risultati ... --gotm ... --uscita ...

--dati  deve contenere il vento ERA5 (scarica_era5.py); --gotm la cartella con le run MATRICE_*,
EXTENSION_* e FINSTAG_* (i loro gotm.yaml sono il punto di partenza).
Scrive in <uscita>/vento_era5/ il pacchetto da estrarre nella cartella del progetto GOTM:
  data/gotm_era5/momentum_flux_era5.dat  sforzo ERA5 (N/m2), riportato a meta' dell'ora su cui e' mediato
  data/gotm_era5/wind10_era5.dat         vento a 10 m ERA5 (m/s), istantaneo, all'ora intera
  templates/<caso>/gotm.yaml              8 casi; nei casi del 4 ottobre la convezione CVMix e' accesa
  scripts/, LEGGIMI_VENTO_ERA5.md         lancio e controlli (da analisi/rimescolamento/vento_era5/)
e lo stesso contenuto in <uscita>/vento_era5_sturla.zip. Istruzioni nel LEGGIMI_VENTO_ERA5.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os
import re
import shutil
import zipfile

import comune as C
import gotm_comune as G

QUI = Path(__file__).resolve().parent
SORGENTI = {"OTT04_R0": "MATRICE_*_R0", "OTT04_R3": "EXTENSION_*_R3_tau6h_hb10m",
            "OTT18_R0": "*_OTTOBRE18_R0", "OTT18_R3": "*_OTTOBRE18_R3",
            "AGO_R0": "*_AGOSTO_R0", "AGO_R3": "*_AGOSTO_R3"}
# (nome del caso, sorgente, vento ERA5?, accendere la convezione?)
CASI = [("OTT04_R0_CMEMS", "OTT04_R0", False, True), ("OTT04_R0_ERA5", "OTT04_R0", True, True),
        ("OTT04_R3_CMEMS", "OTT04_R3", False, True), ("OTT04_R3_ERA5", "OTT04_R3", True, True),
        ("OTTOBRE18_R0_ERA5", "OTT18_R0", True, False), ("OTTOBRE18_R3_ERA5", "OTT18_R3", True, False),
        ("AGOSTO_R0_ERA5", "AGO_R0", True, False), ("AGOSTO_R3_ERA5", "AGO_R3", True, False)]
PERIODO = ("2026-07-01", "2026-10-31 23:00")


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    radice = G.cartella_gotm(a)
    P = os.path.join(a.uscita, "vento_era5")
    shutil.rmtree(P, ignore_errors=True)
    os.makedirs(os.path.join(P, "data", "gotm_era5"))
    src = {k: G.trova_run(radice, s) for k, s in SORGENTI.items()}
    for nome, s, era5, conv in CASI:
        d = src[s]
        y = open(os.path.join(d, "gotm.yaml"), encoding="utf-8").read()
        n0 = len(y)
        y = re.sub(r"^title: .*$", f"title: Sturla 2026 {nome}", y, count=1, flags=re.M)
        if era5:
            y, k1 = re.subn(r"file: \S*/momentum_flux\.dat", "file: ../../data/gotm_era5/momentum_flux_era5.dat", y)
            y, k2 = re.subn(r"file: \S*/wind10\.dat", "file: ../../data/gotm_era5/wind10_era5.dat", y)
            assert (k1, k2) == (2, 2), (nome, k1, k2)
        if conv:
            y, k3 = re.subn(r"(convection:\n\s+use:) false", r"\1 true", y)
            assert k3 == 1, nome
        assert "convection:\n      use: true" in y, f"{nome}: convezione CVMix non accesa"
        c = os.path.join(P, "templates", nome)
        os.makedirs(c)
        with open(os.path.join(c, "gotm.yaml"), "w", newline="\n", encoding="utf-8") as fh:
            fh.write(y)
        if os.path.isfile(os.path.join(d, "init_ts.dat")):
            shutil.copy(os.path.join(d, "init_ts.dat"), c)
        print(f"{nome:28s} da {os.path.basename(d)}  ({n0} -> {len(y)} caratteri)")
    E = C.era5_vento(a.dati).loc[PERIODO[0]:PERIODO[1]]
    tau = C.sforzo_era5_centrato(E)
    with open(os.path.join(P, "data", "gotm_era5", "momentum_flux_era5.dat"), "w", newline="\n") as fh:
        for t, r in tau.iterrows():
            fh.write(f"{t:%Y-%m-%d %H:%M:%S}  {r.tx:.6f}  {r.ty:.6f}\n")
    with open(os.path.join(P, "data", "gotm_era5", "wind10_era5.dat"), "w", newline="\n") as fh:
        for t, r in E[["u10", "v10"]].dropna().iterrows():
            fh.write(f"{t:%Y-%m-%d %H:%M:%S}  {r.u10:.4f}  {r.v10:.4f}\n")
    shutil.copytree(QUI / "vento_era5" / "scripts", os.path.join(P, "scripts"))
    shutil.copy(QUI / "vento_era5" / "LEGGIMI_VENTO_ERA5.md", P)
    zp = os.path.join(a.uscita, "vento_era5_sturla.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(P):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, P))
    print("scritto", zp)


if __name__ == "__main__":
    main()
