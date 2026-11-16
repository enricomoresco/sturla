"""Test delle funzioni comuni degli script di analisi: lettura delle run GOTM, criterio del mixed layer,
vento ERA5 e scelta dei file esterni."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analisi" / "rimescolamento"))
import comune as C  # noqa: E402
import gotm_comune as G  # noqa: E402


def test_mld_T_strato_mescolato_e_colonna_omogenea():
    prof = np.arange(0.0, 35.0, 1.0)
    T = np.where(prof <= 20.0, 24.0, 18.0)                     # salto fra 20 e 21 m
    assert 20.0 < G.mld_T(prof, T) < 21.0
    assert G.mld_T(prof, np.full(prof.shape, 22.0)) == prof[-1]     # mescolata fino al fondo


def _run(cartella, T_fondo, T_sup, z_interfaccia, ore=3):
    """Run GOTM finta: colonna di 34.5 m a 69 livelli, acqua calda sopra z_interfaccia (profondita')."""
    cartella.mkdir(parents=True)
    z = -34.5 + 0.25 + 0.5 * np.arange(69)                     # dal fondo alla superficie, centri delle celle
    T = np.where(-z <= z_interfaccia, T_sup, T_fondo)[None, :].repeat(ore, axis=0)
    t = pd.date_range("2026-10-04", periods=ore, freq="1h")
    ds = xr.Dataset(dict(temp=(("time", "z", "lat", "lon"), T[:, :, None, None]),
                         z=(("time", "z", "lat", "lon"), np.repeat(z[None, :], ore, 0)[:, :, None, None])),
                    coords=dict(time=t))
    ds.to_netcdf(cartella / "sturla.nc")


def test_trova_run_usa_la_piu_recente_e_legge_fondo_e_isoterma(tmp_path):
    _run(tmp_path / "a" / "MATRICE_20261114_100000_R3", 18.0, 24.0, 20.0)
    _run(tmp_path / "b" / "MATRICE_20261114_120000_R3", 18.0, 24.0, 25.0)
    _run(tmp_path / "b" / "MATRICE_20261114_130000_R3_senza_w", 10.0, 12.0, 5.0)
    run = G.trova_run(str(tmp_path), "MATRICE_*_R3")
    assert Path(run).name == "MATRICE_20261114_120000_R3"
    s, dT = G.fondo(run)
    assert np.allclose(s.values, 18.0) and dT == 6.0               # 33.9 m e' nell'acqua fredda
    iso, isoT = G.isoterma(run)
    assert isoT == 21.0 and np.allclose(iso.values, 24.75)        # centro della cella calda piu' profonda
    assert np.allclose(G.mld_run(run).values, 25.0, atol=0.5)


def _era5(path, variabili, tempi):
    lat, lon = [44.5, 44.25, 44.0], [8.75, 9.0, 9.25]
    dv = {v: (("valid_time", "latitude", "longitude"), np.full((len(tempi), 3, 3), val)) for v, val in variabili.items()}
    for v in dv:
        dv[v][1][:, 1, 1] = variabili[v] * 2                      # valore diverso nel punto 44.25 N, 9.0 E
    xr.Dataset(dv, coords=dict(valid_time=tempi, latitude=lat, longitude=lon)).to_netcdf(path)


def test_era5_vento_da_due_file_e_sforzo_centrato(tmp_path):
    t = pd.date_range("2026-10-01", periods=4, freq="1h")
    _era5(tmp_path / "data_stream-oper_stepType-instant.nc", dict(u10=1.0, v10=-2.0, msl=101000.0), t)
    _era5(tmp_path / "data_stream-oper_stepType-avg.nc", dict(avg_iews=0.01, avg_inss=-0.02), t)
    E = C.era5_vento(str(tmp_path))
    assert list(E.columns) == ["u10", "v10", "msl", "tx", "ty"]
    assert np.allclose(E.u10, 2.0) and np.allclose(E.msl, 2020.0) and np.allclose(E.tx, 0.02)
    s = C.sforzo_era5_centrato(E)
    assert s.index[0] == pd.Timestamp("2026-09-30 23:30")


def test_trova_vento_cmems_esclude_il_vento_era5(tmp_path):
    for nome in ("era5_vento_pressione_sturla_2026.nc", "vento_ECMWF_sturla_2026.nc"):
        (tmp_path / nome).write_bytes(b"")
    assert Path(C.trova(str(tmp_path), "vento")).name == "vento_ECMWF_sturla_2026.nc"
