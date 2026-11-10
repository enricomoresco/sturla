"""Prodotti d'archivio: NetCDF CF-1.8 (+ ACDD) e CSV, a partire dai risultati di
``currents`` (currents_qc.nc) e ``waves`` (waves_bulk.nc).

File prodotti in ``<out>/archive/``:
  <ID>_<inizio>_<fine>_currents.nc   profili di corrente + serie al fondo (temperatura, pressione, ...)
  <ID>_<inizio>_<fine>_waves.nc      parametri d'onda per burst + spettri
  <ID>_..._timeseries.csv            serie al fondo + corrente media sulla verticale
  <ID>_..._waves.csv                 parametri d'onda
  <ID>_..._currents_<var>.csv        matrici tempo x quota (u, v, speed, direction)

Flag di qualità (schema IOOS/QARTOD): 1 buono, 2 non valutato, 3 sospetto, 4 cattivo, 9 mancante.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np
import pandas as pd
import xarray as xr


FLAG_ATTRS = dict(flag_values=np.array([1, 2, 3, 4, 9], np.int8),
                  flag_meanings="good not_evaluated suspect bad missing",
                  conventions="IOOS QARTOD")
TIME_ENC = dict(units="seconds since 1970-01-01 00:00:00", calendar="standard", dtype="float64")


def load_metadata(path: str | None) -> dict:
    md = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            md = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    return md


def _global_attrs(md: dict, ds_src: xr.Dataset, kind: str, t: np.ndarray) -> dict:
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    a = {
        "Conventions": "CF-1.8, ACDD-1.3",
        "featureType": "timeSeriesProfile" if kind == "currents" else "timeSeries",
        "title": md.get("title") or f"ADCP {kind}",
        "summary": ("Profili di corrente mediati per burst da ping singoli (Teledyne RDI Sentinel V), "
                    "temperatura e pressione al fondo" if kind == "currents" else
                    "Parametri d'onda per burst da tracciamento della superficie con il beam verticale "
                    "e da pressione; direzione con metodo PUV (superficie-velocita')"),
        "station_id": md.get("station_id") or "ADCP",
        "platform": "struttura sul fondale, strumento rivolto verso l'alto",
        "instrument": "Teledyne RDI Sentinel V",
        "instrument_serial_number": str(ds_src.attrs.get("serial", "")),
        "instrument_firmware": f"{ds_src.attrs.get('fw_ver', '')}.{ds_src.attrs.get('fw_rev', '')}",
        "time_coverage_start": str(pd.Timestamp(t.min())) if len(t) else "",
        "time_coverage_end": str(pd.Timestamp(t.max())) if len(t) else "",
        "date_created": now,
        "history": f"{now}: creato con adcp_pd0 (preprocessing PD0 -> archivio)",
        "processing_level": "dati controllati automaticamente (QC), non validati manualmente",
        "source_file": str(ds_src.attrs.get("source_file", "")),
    }
    for k in ("deployment_start", "deployment_end", "deployment_method"):
        if ds_src.attrs.get(k):
            a[k] = str(ds_src.attrs[k])
    for k in ("instrument_height_above_bed_m", "institution", "project", "creator_name", "creator_email", "license", "deployment_id",
              "deployment_start", "deployment_end", "time_zone", "comment",
              "geospatial_vertical_reference", "magnetic_declination_deg", "magnetic_declination_applied"):
        if md.get(k) is not None:
            v = md[k]
            a[k] = str(v) if isinstance(v, bool) else v
    if md.get("latitude") is not None and md.get("longitude") is not None:
        a.update(geospatial_lat_min=md["latitude"], geospatial_lat_max=md["latitude"],
                 geospatial_lon_min=md["longitude"], geospatial_lon_max=md["longitude"])
    # parametri di elaborazione (prefisso proc_)
    for k, v in ds_src.attrs.items():
        if k.startswith(("qc_", "avg_")) or k in ("hab", "p_atm", "nperseg", "fmin", "fmax", "kp2_min",
                                                    "declination", "fs", "n_short_bursts_skipped"):
            a[f"proc_{k}"] = v if not isinstance(v, (bool, np.bool_)) else int(v)
    return a


def _station_coords(ds, md):
    if md.get("latitude") is not None and md.get("longitude") is not None:
        ds = ds.assign_coords(
            latitude=((), float(md["latitude"]), dict(standard_name="latitude", units="degrees_north")),
            longitude=((), float(md["longitude"]), dict(standard_name="longitude", units="degrees_east")))
    ds["station"] = ((), str(md.get("station_id") or "ADCP"), dict(cf_role="timeseries_id"))
    return ds


def currents_archive(qc: xr.Dataset, md: dict, hab: float) -> xr.Dataset:
    t = qc.time.values
    z = qc.range.values.astype(np.float32) + np.float32(hab)
    dims = ("time", "z")
    # flag per cella: 4 se un test QC fallisce, 9 se manca senza flag, 1 altrimenti
    any_flag = np.zeros(qc.u.shape, bool)
    for k in [k for k in qc.data_vars if k.startswith("flag_")]:
        any_flag |= qc[k].values.astype(bool)
    valid = np.isfinite(qc.u.values)
    flag = np.where(valid, 1, np.where(any_flag, 4, 9)).astype(np.int8)
    if "good_ping_frac" in qc:
        gp = qc.good_ping_frac.values
        flag = np.where(valid & (gp < 0.8), 3, flag).astype(np.int8)
    V = lambda k: qc[k].values.astype(np.float32)
    ds = xr.Dataset(coords=dict(
        time=("time", t, dict(standard_name="time", long_name="tempo (centro del burst)", axis="T")),
        z=("z", z, dict(standard_name="height_above_sea_floor", long_name="quota della cella sul fondale",
                        units="m", positive="up", axis="Z"))))
    vel = dict(u=("eastward_sea_water_velocity", "velocita' verso Est"),
               v=("northward_sea_water_velocity", "velocita' verso Nord"),
               w=("upward_sea_water_velocity", "velocita' verticale"))
    for k, (sn, ln) in vel.items():
        ds[k] = (dims, V(k), dict(standard_name=sn, long_name=ln, units="m s-1",
                                  ancillary_variables="vel_qc"))
    ds["err"] = (dims, V("err"), dict(long_name="velocita' d'errore (4 beam)", units="m s-1"))
    ds["speed"] = (dims, V("speed"), dict(standard_name="sea_water_speed", long_name="intensita' corrente",
                                          units="m s-1", ancillary_variables="vel_qc"))
    ds["direction"] = (dims, V("direction"), dict(standard_name="direction_of_sea_water_velocity",
                                                  long_name="direzione verso cui va la corrente, da Nord orario",
                                                  units="degree", ancillary_variables="vel_qc"))
    if "echo_mean" in qc:
        ds["echo"] = (dims, V("echo_mean"), dict(long_name="intensita' eco media sui 4 beam", units="count"))
    ds["vel_qc"] = (dims, flag, dict(long_name="flag di qualita' delle velocita'", **FLAG_ATTRS))
    if "in_position" in qc:
        dq = np.where(qc.in_position.values.astype(bool), 1, 4).astype(np.int8)
    else:
        dq = np.full(len(t), 2, np.int8)
    ds["deployment_qc"] = ("time", dq, dict(
        long_name="flag strumento in posizione sul fondo",
        comment="4 = fuori posizione (in aria, discesa, recupero): tutti i dati di quell'istante sono da scartare",
        **FLAG_ATTRS))
    ts = dict(
        temperature=("sea_water_temperature", "temperatura dell'acqua al fondo (sensore dello strumento)", "degree_Celsius"),
        pressure=("sea_water_pressure", "pressione al sensore (media sul burst)", "dbar"),
        heading=("platform_orientation", "heading (media circolare sul burst)", "degree"),
        pitch=("platform_pitch", "pitch", "degree"),
        roll=("platform_roll", "roll", "degree"),
    )
    for k, (sn, ln, u) in ts.items():
        if k in qc:
            ds[k] = ("time", V(k), dict(standard_name=sn, long_name=ln, units=u,
                                        ancillary_variables="deployment_qc"))
    if "pressure" in qc:
        d = qc.pressure.values * 1e4 / (1025.0 * 9.81) + hab
        ds["water_depth"] = ("time", d.astype(np.float32), dict(
            standard_name="sea_floor_depth_below_sea_surface", units="m",
            long_name="profondita' totale (pressione/(rho g) + quota sensore)",
            comment="rho = 1025 kg m-3, g = 9.81 m s-2; pressione assunta relativa (azzerata in aria)"))
    if "pressure_std" in qc:
        ds["pressure_std"] = ("time", V("pressure_std"), dict(long_name="dev. std della pressione nel burst",
                                                             units="dbar"))
    if "n_pings" in qc:
        ds["n_pings"] = ("time", qc.n_pings.values.astype(np.int32),
                         dict(long_name="numero di ping mediati", units="1"))
    ds["u_davg"] = ("time", V("u_davg"), dict(long_name="velocita' Est media sulla verticale", units="m s-1"))
    ds["v_davg"] = ("time", V("v_davg"), dict(long_name="velocita' Nord media sulla verticale", units="m s-1"))
    ds["speed_davg"] = ("time", V("speed_davg"), dict(long_name="intensita' media sulla verticale", units="m s-1"))
    ds["dir_davg"] = ("time", V("dir_davg"), dict(long_name="direzione (verso cui va) media sulla verticale",
                                                  units="degree"))
    ds = _station_coords(ds, md)
    ds.attrs = _global_attrs(md, qc, "currents", t)
    ds.attrs["comment_temperature"] = ("temperatura misurata dal sensore nella testa dello strumento "
                                       "(al fondo); la salinita' non e' misurata (valore impostato)")
    return ds


def waves_archive(w: xr.Dataset, md: dict, hm0_dir_min: float = 0.25) -> xr.Dataset:
    t = w.time.values
    has_vb = "Hm0_vb" in w and np.isfinite(w.Hm0_vb).any()
    sfx = "_vb" if has_vb else ""
    method = ("superficie tracciata con l'eco del beam verticale" if has_vb else
              "pressione corretta con la teoria lineare")
    ds = xr.Dataset(coords=dict(
        time=("time", t, dict(standard_name="time", long_name="tempo (inizio del burst)", axis="T")),
        frequency=("frequency", w.freq.values.astype(np.float32),
                   dict(standard_name="sea_surface_wave_frequency", units="Hz"))))
    par = dict(
        Hm0=("sea_surface_wave_significant_height", "altezza significativa spettrale Hm0", "m"),
        Tp=("sea_surface_wave_period_at_variance_spectral_density_maximum", "periodo di picco", "s"),
        Tm01=("sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
              "periodo medio Tm01", "s"),
        Tm02=("sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment",
              "periodo medio Tm02", "s"),
    )
    for k, (sn, ln, u) in par.items():
        src = k + sfx
        if src in w:
            ds[k] = ("time", w[src].values.astype(np.float32),
                     dict(standard_name=sn, long_name=f"{ln} ({method})", units=u, ancillary_variables="wave_qc"))
    # flag
    hm = ds["Hm0"].values if "Hm0" in ds else np.full(len(t), np.nan)
    flag = np.where(np.isfinite(hm), 1, 9).astype(np.int8)
    if has_vb and "surface_found_frac" in w:
        flag = np.where(np.isfinite(hm) & (w.surface_found_frac.values < 0.9), 3, flag).astype(np.int8)
    if "in_position" in w:
        flag = np.where(w.in_position.values.astype(bool), flag, 4).astype(np.int8)
    ds["wave_qc"] = ("time", flag, dict(long_name="flag di qualita' dei parametri d'onda",
                                        comment="3 = superficie trovata in meno del 90% dei ping", **FLAG_ATTRS))
    if "Dp" in w:
        ds["Dp"] = ("time", w.Dp.values.astype(np.float32), dict(
            standard_name="sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
            long_name="direzione di provenienza al picco, da Nord orario", units="degree",
            ancillary_variables="direction_qc"))
        ds["Dm"] = ("time", w.Dm.values.astype(np.float32), dict(
            standard_name="sea_surface_wave_from_direction",
            long_name="direzione media di provenienza, da Nord orario", units="degree",
            ancillary_variables="direction_qc"))
        dflag = np.where(np.isfinite(w.Dp.values), np.where(hm < hm0_dir_min, 3, 1), 9).astype(np.int8)
        if "in_position" in w:
            dflag = np.where(w.in_position.values.astype(bool), dflag, 4).astype(np.int8)
        ds["direction_qc"] = ("time", dflag, dict(long_name="flag di qualita' delle direzioni",
                                                   comment=f"3 = Hm0 < {hm0_dir_min} m (poca energia)", **FLAG_ATTRS))
    if has_vb:
        ds["Hm0_pressure"] = ("time", w.Hm0.values.astype(np.float32), dict(
            long_name="Hm0 da pressione (solo frequenze sotto la frequenza di taglio)", units="m",
            comment="in acqua profonda sottostima: non usare come Hm0"))
        ds["S"] = (("time", "frequency"), w.S_vb.values.astype(np.float32), dict(
            standard_name="sea_surface_wave_variance_spectral_density", units="m2 Hz-1",
            long_name="spettro di elevazione (beam verticale)"))
        ds["surface_found_frac"] = ("time", w.surface_found_frac.values.astype(np.float32),
                                    dict(long_name="frazione di ping con superficie trovata", units="1"))
    ds["S_pressure"] = (("time", "frequency"), w.S_p.values.astype(np.float32), dict(
        units="m2 Hz-1", long_name="spettro di elevazione da pressione (fino alla frequenza di taglio)"))
    ds["fcut_pressure"] = ("time", w.fcut.values.astype(np.float32),
                           dict(long_name="frequenza di taglio per la pressione", units="Hz"))
    ds["water_depth"] = ("time", w.depth.values.astype(np.float32), dict(
        standard_name="sea_floor_depth_below_sea_surface", units="m", long_name="profondita' media nel burst"))
    ds["n_samples"] = ("time", w.n.values.astype(np.int32), dict(long_name="campioni nel burst", units="1"))
    ds = _station_coords(ds, md)
    ds.attrs = _global_attrs(md, w, "waves", t)
    ds.attrs["wave_method"] = method
    return ds


def _encoding(ds):
    enc = {}
    for k, v in ds.data_vars.items():
        if v.dtype.kind in "fiu":
            e = dict(zlib=True, complevel=4)
            if v.dtype.kind == "f":
                e["_FillValue"] = np.float32(-9999.0) if v.dtype == np.float32 else -9999.0
            enc[k] = e
    enc["time"] = TIME_ENC
    return enc


def _csv_header(ds, extra=""):
    a = ds.attrs
    lines = [f"# {a.get('title', '')}", f"# stazione: {a.get('station_id', '')}",
             f"# strumento: {a.get('instrument', '')} s/n {a.get('instrument_serial_number', '')}",
             f"# periodo: {a.get('time_coverage_start', '')} - {a.get('time_coverage_end', '')}",
             f"# fuso orario: {a.get('time_zone', 'UTC (da verificare)')}"]
    if "geospatial_lat_min" in a:
        lines.append(f"# lat/lon: {a['geospatial_lat_min']} / {a['geospatial_lon_min']}")
    lines.append("# flag: 1 buono, 2 non valutato, 3 sospetto, 4 cattivo, 9 mancante; valori mancanti vuoti")
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n"


def _write_csv(path, df, header):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        df.to_csv(fh, index=True, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.4f")


def write_archive(outdir: str, md: dict, hab: float, verbose: bool = True) -> list:
    mh = md.get("instrument_height_above_bed_m")
    if mh is not None and abs(float(mh) - hab) > 1e-6:
        print(f"ATTENZIONE: --hab {hab} diverso da instrument_height_above_bed_m={mh} nei metadati: "
              f"si usa --hab (lo stesso usato per le onde)")
    md = dict(md, instrument_height_above_bed_m=hab)
    adir = os.path.join(outdir, "archive")
    os.makedirs(adir, exist_ok=True)
    written = []
    sid = (md.get("station_id") or "ADCP").replace(" ", "_")
    fmt = lambda t: pd.Timestamp(t).strftime("%Y%m%d")
    pc, pw = os.path.join(outdir, "currents_qc.nc"), os.path.join(outdir, "waves_bulk.nc")
    if os.path.exists(pc):
        qc = xr.load_dataset(pc)
        c = currents_archive(qc, md, hab)
        base = os.path.join(adir, f"{sid}_{fmt(c.time.values.min())}_{fmt(c.time.values.max())}")
        c.to_netcdf(base + "_currents.nc", encoding=_encoding(c))
        written.append(base + "_currents.nc")
        cols = [k for k in ("deployment_qc", "temperature", "pressure", "water_depth", "heading", "pitch", "roll", "n_pings",
                            "u_davg", "v_davg", "speed_davg", "dir_davg") if k in c]
        df = c[cols].to_dataframe()[cols]
        units = ", ".join(f"{k} [{c[k].attrs.get('units', '')}]" for k in cols)
        _write_csv(base + "_timeseries.csv", df, _csv_header(c, f"# colonne: {units}"))
        written.append(base + "_timeseries.csv")
        for k in ("u", "v", "speed", "direction", "vel_qc"):
            m = pd.DataFrame(c[k].values, index=pd.DatetimeIndex(c.time.values, name="time"),
                             columns=[f"z{z:.2f}m" for z in c.z.values])
            _write_csv(base + f"_currents_{k}.csv", m,
                       _csv_header(c, f"# {k} [{c[k].attrs.get('units', 'flag')}]; colonne = quota sul fondale"))
            written.append(base + f"_currents_{k}.csv")
    if os.path.exists(pw):
        w = xr.load_dataset(pw)
        wa = waves_archive(w, md)
        base = os.path.join(adir, f"{sid}_{fmt(wa.time.values.min())}_{fmt(wa.time.values.max())}")
        wa.to_netcdf(base + "_waves.nc", encoding=_encoding(wa))
        written.append(base + "_waves.nc")
        cols = [k for k in ("Hm0", "Tp", "Tm01", "Tm02", "Dp", "Dm", "wave_qc", "direction_qc",
                            "water_depth", "Hm0_pressure", "n_samples") if k in wa]
        df = wa[cols].to_dataframe()[cols]
        units = ", ".join(f"{k} [{wa[k].attrs.get('units', 'flag')}]" for k in cols)
        _write_csv(base + "_waves.csv", df, _csv_header(wa, f"# metodo: {wa.attrs['wave_method']}\n# colonne: {units}"))
        written.append(base + "_waves.csv")
    if verbose:
        print("\nArchivio:")
        for f in written:
            print(f"  {f}")
    return written
