"""Impostazioni e funzioni comuni dell'analisi del rimescolamento alla stazione di Sturla.

Tutti gli script di questa cartella accettano gli stessi tre argomenti:

  --dati        cartella con i file esterni (ERA5, vento CMEMS, modello MedFS)
  --risultati   cartella dei risultati dell'ADCP (con dentro preprocessing/ e postprocessing/)
  --uscita      cartella dove scrivere tabelle e figure (default: out_rimescolamento)

I valori legati alla stazione e al periodo sono qui sotto: per un altro deployment vanno
cambiati qui.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

QUI = Path(__file__).resolve().parent
RADICE = QUI.parent.parent
for p in (str(QUI), str(RADICE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ------------------------------------------------------------------ costanti fisiche
g, RHO0, CP, RHOA = 9.81, 1025.0, 3990.0, 1.22

# ------------------------------------------------------------------ stazione e periodo
H = 34.5                                    # fondale alla stazione (m)
ALTEZZA_SENSORE = 0.6                       # sensore di temperatura dell'ADCP sopra il fondale (m)
PROF_SENSORE = H - ALTEZZA_SENSORE          # profondita' del sensore (m)
LAT_STAZIONE = 44.38
FCOR = 2 * 7.2921e-5 * np.sin(np.deg2rad(LAT_STAZIONE))
PUNTO_MODELLO = dict(latitude=44.3542, longitude=9.0)    # punto MedFS a mare della stazione
PUNTO_ERA5 = dict(latitude=44.25, longitude=9.0)         # la riga a 44.5 N e' terra
VENTO_LAT, VENTO_LON = 44.3125, (8.9375, 9.0625)         # riga a mare del vento CMEMS: media dei due punti
ASSE_COSTA, VERSO_COSTA = 281.0, 11.0       # gradi da nord: lungo costa (verso ONO) e verso costa
PERIODO = ("2026-07-10", "2026-10-27 15:00")              # in posizione
EVENTI = {"21-22 agosto": ("2026-08-19 12:00", "2026-08-23 12:00"),
          "5-9 ottobre": ("2026-10-04 12:00", "2026-10-09 12:00"),
          "19-22 ottobre": ("2026-10-18 12:00", "2026-10-22 12:00")}
FINESTRE = [("21-22 agosto", "2026-08-19 12:00", "2026-08-23 12:00", "21-22 agosto"),
            ("5-9 ottobre", "2026-10-04 12:00", "2026-10-09 12:00", "5-9 ottobre"),
            ("prima della mareggiata", "2026-10-04 12:00", "2026-10-07 12:00", "5-9 ottobre"),
            ("durante la mareggiata", "2026-10-07 12:00", "2026-10-09 12:00", "5-9 ottobre"),
            ("19-22 ottobre", "2026-10-18 12:00", "2026-10-22 12:00", "19-22 ottobre")]

# file esterni: il nome puo' avere un prefisso qualsiasi
FILE = {"flussi": "*flussi_ERA5*.nc", "vento": "*vento*.nc", "temperatura": "*temperatura_3D*.nc",
        "salinita": "*salinita_3D*.nc", "mld": "*mixed_layer_depth*.nc"}


def argomenti(descrizione: str, argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=descrizione)
    ap.add_argument("--dati", required=True, help="cartella con i file esterni (ERA5, vento, MedFS)")
    ap.add_argument("--risultati", required=True, help="cartella dei risultati ADCP (preprocessing/, postprocessing/)")
    ap.add_argument("--uscita", default="out_rimescolamento", help="cartella di uscita")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.uscita, "figure"), exist_ok=True)
    return a


def trova(cartella: str, chiave: str) -> str:
    """Percorso del file esterno indicato da `chiave` (vedi FILE) o di un file dei risultati."""
    pat = FILE.get(chiave, chiave)
    hit = sorted(glob.glob(os.path.join(cartella, pat))) or sorted(glob.glob(os.path.join(cartella, "**", pat), recursive=True))
    if not hit:
        raise FileNotFoundError(f"nessun file '{pat}' in {cartella}")
    if len(hit) > 1:
        print(f"  attenzione: piu' file per '{pat}', uso {hit[0]}")
    return hit[0]


def rho_eos(t, s):
    """Densita' dell'acqua di mare a pressione zero (EOS-80), kg/m3."""
    rw = 999.842594 + 6.793952e-2*t - 9.095290e-3*t**2 + 1.001685e-4*t**3 - 1.120083e-6*t**4 + 6.536332e-9*t**5
    return (rw + s*(0.824493 - 4.0899e-3*t + 7.6438e-5*t**2 - 8.2467e-7*t**3 + 5.3875e-9*t**4)
            + s**1.5*(-5.72466e-3 + 1.0227e-4*t - 1.6546e-6*t**2) + 4.8314e-4*s**2)


def modello_punto(dati: str):
    """Temperatura e salinita' orarie del modello nel punto a mare della stazione.
    Restituisce (profondita', tempi, T[tempo, quota], S[tempo, quota])."""
    T = xr.open_dataset(trova(dati, "temperatura")).thetao.sel(PUNTO_MODELLO, method="nearest")
    S = xr.open_dataset(trova(dati, "salinita")).so.sel(PUNTO_MODELLO, method="nearest")
    return (T.depth.values.astype(float), pd.DatetimeIndex(T.time.values),
            T.values.astype(float), S.values.astype(float))


def mld_modello(dati: str) -> pd.Series:
    M = xr.open_dataset(trova(dati, "mld")).mlotst.sel(PUNTO_MODELLO, method="nearest").to_series()
    M.index = pd.DatetimeIndex(M.index)
    return M


def serie_orarie(risultati: str) -> pd.DataFrame:
    """Serie orarie dell'ADCP, con l'orario riportato all'inizio dell'ora (come i forzanti)."""
    h = pd.read_csv(trova(risultati, "serie_orarie.csv"), index_col=0, parse_dates=True)
    h.index = h.index - pd.Timedelta("30min")
    return h


def forzanti_1d(uscita: str, risultati: str) -> pd.DataFrame:
    """Forzanti orari per il modello 1D: quelli di forzanti.py piu' la corrente media misurata;
    i buchi brevi sono interpolati, le onde mancanti valgono zero."""
    F = leggi(uscita, "forzanti_orari.csv")
    h = serie_orarie(risultati)
    v = h.corrente_media_vel.where(h.deployment_qc == 1)
    F["Umean"] = v.reindex(F.index).interpolate(limit=6).fillna(v.median())
    for c in ("Us0", "Hm0", "Ub"):
        F[c] = F[c].interpolate(limit=6).fillna(0.0)
    return F


def leggi(uscita: str, nome: str, **kw) -> pd.DataFrame:
    p = os.path.join(uscita, nome)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} non esiste: lanciare prima il passo che lo produce (vedi esegui_tutto.py)")
    return pd.read_csv(p, index_col=0, parse_dates=True, **kw)
