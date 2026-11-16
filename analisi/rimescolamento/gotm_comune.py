"""Funzioni comuni per leggere le run GOTM di Sturla (cartelle con gotm.yaml e sturla.nc).

Una run si trova cercando, sotto la cartella indicata con --gotm (a qualunque profondita'), la
cartella il cui nome corrisponde allo schema del caso: per esempio MATRICE_*_R3,
EXTENSION_*_R3_tau6h_hb10m, *_OTTOBRE18_R0 o VENTO_*_OTT04_R0_CMEMS. Se lo stesso caso e' stato
lanciato piu' volte si usa la run piu' recente: il nome della cartella comincia con data e ora del
lancio, quindi e' l'ultima in ordine alfabetico.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

import comune as C

PROF_SENSORE = 33.9          # profondita' del sensore di temperatura dell'ADCP nelle run (34.5 - 0.6 m)


def cartella_gotm(a) -> str:
    if not getattr(a, "gotm", None):
        raise SystemExit("serve --gotm: la cartella con le run GOTM (cartelle dei casi con sturla.nc)")
    return a.gotm


def trova_run(radice: str, schema: str) -> str:
    """Cartella della run piu' recente il cui nome corrisponde a `schema` (glob) e contiene sturla.nc."""
    hit = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(radice, "**", schema, "sturla.nc"),
                                                         recursive=True)}, key=os.path.basename)
    if not hit:
        raise FileNotFoundError(f"nessuna run '{schema}' con sturla.nc sotto {radice}")
    if len(hit) > 1:
        print(f"  {schema}: {len(hit)} run, uso la piu' recente {os.path.basename(hit[-1])}")
    return hit[-1]


def profili(run: str):
    """Tempi, quote (m, negative, dal fondo alla superficie) e temperatura [tempo, quota] di una run."""
    with xr.open_dataset(os.path.join(run, "sturla.nc")) as d:
        t = pd.DatetimeIndex(d.time.values)
        z = d.z.squeeze().isel(time=0).values.astype(float)
        T = d.temp.squeeze().values.astype(float)
    return t, z, T


def fondo(run: str, prof: float = PROF_SENSORE):
    """Temperatura della run alla profondita' del sensore, ora per ora, e differenza superficie - fondo
    del profilo iniziale."""
    t, z, T = profili(run)
    s = pd.Series([np.interp(-prof, z, T[i]) for i in range(len(t))], index=t)
    return s, float(T[0, -1] - T[0, 0])


def isoterma(run: str):
    """Profondita' (m, positiva) dell'isoterma media fra superficie e fondo del profilo iniziale: per ogni
    ora, il livello piu' basso con temperatura almeno pari all'isoterma. Restituisce (serie, isoterma)."""
    t, z, T = profili(run)
    iso = 0.5 * (T[0, 0] + T[0, -1])
    zi = []
    for i in range(len(t)):
        k = np.flatnonzero(T[i] >= iso)
        zi.append(-z[k[0]] if len(k) and k[0] > 0 else np.nan)
    return pd.Series(zi, index=t), float(iso)


def mld_T(prof, T, dT: float = 0.2, zref: float = 10.0) -> float:
    """Profondita' (m, positiva) dove la temperatura scende di `dT` sotto il valore a `zref` m; se non
    succede, la profondita' massima del profilo (colonna mescolata fino al fondo)."""
    prof, T = np.asarray(prof, float), np.asarray(T, float)
    o = np.argsort(prof)
    prof, T = prof[o], T[o]
    ok = np.isfinite(T)
    prof, T = prof[ok], T[ok]
    if not len(T):
        return np.nan
    tr = np.interp(zref, prof, T) - dT
    sotto = np.flatnonzero((prof > zref) & (T <= tr))
    if not len(sotto):
        return float(prof[-1])
    i = sotto[0]
    return float(np.interp(tr, [T[i], T[i - 1]], [prof[i], prof[i - 1]]))


def mld_run(run: str) -> pd.Series:
    """Profondita' del mixed layer di una run, ora per ora, con il criterio di mld_T."""
    t, z, T = profili(run)
    return pd.Series([mld_T(-z, T[i]) for i in range(len(t))], index=t)


def leggi_dat(p: str) -> pd.DataFrame:
    """File di ingresso GOTM con data, ora e due colonne (x, y)."""
    m = pd.read_csv(p, sep=r"\s+", header=None, comment="#").iloc[:, :4]
    m.columns = ["d", "t", "x", "y"]
    m.index = pd.to_datetime(m.d + " " + m.t)
    return m[["x", "y"]]


def trova_dat(radice: str, nome: str, preferisci: str = "") -> str:
    """File di ingresso GOTM `nome` sotto `radice`; se ce n'e' piu' d'uno, quello con `preferisci` nel
    percorso, altrimenti il primo in ordine alfabetico."""
    hit = sorted(glob.glob(os.path.join(radice, "**", nome), recursive=True))
    if not hit:
        raise FileNotFoundError(f"nessun file {nome} sotto {radice}")
    pref = [h for h in hit if preferisci and preferisci in h]
    return (pref or hit)[0]


def temperatura_adcp(risultati: str) -> pd.Series:
    """Temperatura oraria al fondo misurata dall'ADCP, solo in posizione, orario all'inizio dell'ora."""
    h = C.serie_orarie(risultati)
    return h.temperatura.where(h.deployment_qc == 1)
