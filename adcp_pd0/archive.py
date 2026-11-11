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


# --- flag con causa ---------------------------------------------------------------------
# Ogni prodotto ha un flag principale (QARTOD: 1/3/4/9) e una variabile "causa" in cui ogni
# bit indica un motivo diverso; il flag principale è il più grave fra le cause presenti.
VEL_CAUSES = [  # (bit, nome, flag)
    (1, "fuori_posizione", 4), (2, "movimento_telaio", 3), (4, "contaminazione_superficie", 4),
    (8, "velocita_errore_alta", 4), (16, "inclinazione_eccessiva", 4), (32, "pochi_ping_validi", 4),
    (64, "ping_validi_ridotti", 3), (128, "correlazione_bassa", 4), (256, "percent_good_basso", 4),
    (512, "nessun_dato", 9), (1024, "valore_mancante", 9)]
WAVE_CAUSES = [(1, "fuori_posizione", 4), (2, "movimento_telaio", 3), (4, "superficie_trovata_parzialmente", 3),
               (8, "nessun_dato", 9)]
DIR_CAUSES = [(1, "fuori_posizione", 4), (2, "movimento_telaio", 3), (4, "bassa_energia", 3),
              (8, "nessun_dato", 9)]
TEMP_CAUSES = [(1, "fuori_posizione", 4), (2, "pochi_campioni", 3), (4, "nessun_dato", 9)]
POS_CAUSES = [(1, "fuori_posizione", 4), (2, "nessun_dato", 9)]
CAUSE_DESCRIPTIONS = {
    "fuori_posizione": "strumento non fermo sul fondo (in aria, discesa, recupero): dato da scartare",
    "movimento_telaio": "il telaio stava ruotando/spostandosi (salto di heading o inclinazione): dato sospetto",
    "contaminazione_superficie": "cella troppo vicina alla superficie (contaminazione side-lobe): dato rimosso",
    "velocita_errore_alta": "velocita' d'errore oltre la soglia: dato rimosso",
    "inclinazione_eccessiva": "inclinazione dello strumento oltre la soglia: dato rimosso",
    "pochi_ping_validi": "meno del 50% di ping validi (correlazione bassa) nell'intervallo: dato rimosso",
    "ping_validi_ridotti": "fra 50% e 80% di ping validi nell'intervallo: dato sospetto",
    "correlazione_bassa": "correlazione sotto la soglia: dato rimosso",
    "percent_good_basso": "percent good sotto la soglia: dato rimosso",
    "nessun_dato": "nessuna misura nell'intervallo",
    "valore_mancante": "valore non disponibile senza una causa specifica",
    "superficie_trovata_parzialmente": "superficie trovata in meno del 90% dei ping del burst: dato sospetto",
    "bassa_energia": "Hm0 sotto la soglia: direzione poco affidabile, dato sospetto",
    "pochi_campioni": "meno di 30 s di misure nell'ora: dato sospetto",
}


def combine_flags(shape, conditions, causes):
    """conditions: {nome: maschera}; restituisce (flag int8, causa uint16)."""
    flag = np.ones(shape, np.int8)
    cause = np.zeros(shape, np.uint16)
    for bit, name, sev in causes:
        m = conditions.get(name)
        if m is None:
            continue
        m = np.broadcast_to(np.asarray(m, bool), shape)
        cause |= np.where(m, bit, 0).astype(np.uint16)
        flag = np.maximum(flag, np.where(m, sev, 1)).astype(np.int8)
    return flag, cause


def cause_attrs(causes, what):
    return dict(long_name=f"causa del flag di {what} (somma di bit)",
                flag_masks=np.array([c[0] for c in causes], np.uint16),
                flag_meanings=" ".join(c[1] for c in causes),
                comment="; ".join(f"{c[0]} = {c[1]} (flag {c[2]})" for c in causes))


def cause_text(bits, causes):
    bits = np.asarray(bits).astype(int)
    return ["+".join(name for bit, name, _ in causes if b & bit) for b in bits]


def flag_stats(flag, cause, causes) -> dict:
    """Conteggi per valore di flag e per causa."""
    f = np.asarray(flag).ravel()
    c = np.asarray(cause).ravel().astype(int)
    n = f.size
    return dict(n=int(n), per_flag={int(v): int((f == v).sum()) for v in (1, 3, 4, 9)},
                per_causa={name: int(((c & bit) > 0).sum()) for bit, name, _ in causes})


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
    shape = qc.u.shape
    F = lambda k: qc[k].values.astype(bool) if k in qc else None
    n_p = qc.n_pings.values if "n_pings" in qc else None
    empty = (n_p == 0) if n_p is not None else np.zeros(shape[0], bool)
    pos_bad = F("flag_position")
    if pos_bad is None and "in_position" in qc:
        pos_bad = ~qc.in_position.values.astype(bool)[:, None]
    gp = qc.good_ping_frac.values if "good_ping_frac" in qc else None
    min_gp = float(qc.attrs.get("avg_min_good_frac", 0.5))
    cond = {
        "fuori_posizione": pos_bad,
        "movimento_telaio": F("flag_motion")[:, None] if "flag_motion" in qc else None,
        "contaminazione_superficie": F("flag_sidelobe"),
        "velocita_errore_alta": F("flag_err"),
        "inclinazione_eccessiva": F("flag_tilt"),
        "pochi_ping_validi": (gp < min_gp) if gp is not None else None,
        "ping_validi_ridotti": ((gp >= min_gp) & (gp < 0.8)) if gp is not None else None,
        "correlazione_bassa": F("flag_corr"),
        "percent_good_basso": F("flag_pg"),
    }
    flag, cause = combine_flags(shape, cond, VEL_CAUSES)
    valid = np.isfinite(qc.u.values)
    miss = ~valid & (flag == 1)
    flag[miss] = 9
    cause[miss] |= 1024
    flag[empty] = 9
    cause[empty] = 512
    V = lambda k: qc[k].values.astype(np.float32)
    ds = xr.Dataset(coords=dict(
        time=("time", t, dict(standard_name="time", axis="T", long_name=(
            f"centro dell'intervallo di media di {qc.attrs['avg_bin_minutes']:g} min"
            if qc.attrs.get("avg_bin_minutes") else "centro del burst"))),
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
    ds["vel_qc"] = (dims, flag, dict(long_name="flag di qualita' delle velocita'",
                                     ancillary_variables="vel_qc_cause", **FLAG_ATTRS))
    ds["vel_qc_cause"] = (dims, cause, cause_attrs(VEL_CAUSES, "velocita'"))
    inpos = qc.in_position.values.astype(bool) if "in_position" in qc else np.ones(len(t), bool)
    dq, dc = combine_flags((len(t),), {"fuori_posizione": ~inpos & ~empty, "nessun_dato": empty}, POS_CAUSES)
    ds["deployment_qc"] = ("time", dq, dict(
        long_name="flag strumento in posizione sul fondo (vale per temperatura, pressione, assetto)",
        ancillary_variables="deployment_qc_cause", **FLAG_ATTRS))
    ds["deployment_qc_cause"] = ("time", dc, cause_attrs(POS_CAUSES, "posizione"))
    if "flag_motion" in qc:
        ds["platform_motion"] = ("time", qc.flag_motion.values.astype(np.int8), dict(
            long_name="1 = telaio in movimento nell'intervallo (rotazione/inclinazione)", units="1"))
    ts = dict(
        temperature=("sea_water_temperature", "temperatura dell'acqua al fondo (sensore dello strumento)", "degree_Celsius"),
        pressure=("sea_water_pressure", "pressione al sensore (media nell'intervallo)", "dbar"),
        heading=("platform_orientation", "heading (media circolare nell'intervallo)", "degree"),
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
        ds["pressure_std"] = ("time", V("pressure_std"), dict(long_name="dev. std della pressione nell'intervallo",
                                                             units="dbar"))
    if "n_pings" in qc:
        ds["n_pings"] = ("time", qc.n_pings.values.astype(np.int32),
                         dict(long_name="numero di ping mediati", units="1"))
    if "in_position_frac" in qc:
        ds["in_position_frac"] = ("time", qc.in_position_frac.values.astype(np.float32), dict(
            long_name="frazione di ping in posizione nell'intervallo", units="1",
            comment="se > 0 la media usa solo i ping in posizione"))
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
    # flag con causa
    hm = ds["Hm0"].values if "Hm0" in ds else np.full(len(t), np.nan)
    n = len(t)
    inpos = w.in_position.values.astype(bool) if "in_position" in w else np.ones(n, bool)
    mot = w.flag_motion.values.astype(bool) if "flag_motion" in w else np.zeros(n, bool)
    sf = w.surface_found_frac.values if (has_vb and "surface_found_frac" in w) else np.ones(n)
    nod = ~np.isfinite(hm) & inpos
    flag, cause = combine_flags((n,), {"fuori_posizione": ~inpos, "movimento_telaio": mot & inpos,
                                       "superficie_trovata_parzialmente": inpos & np.isfinite(hm) & (sf < 0.9),
                                       "nessun_dato": nod}, WAVE_CAUSES)
    ds["wave_qc"] = ("time", flag, dict(long_name="flag di qualita' dei parametri d'onda",
                                        ancillary_variables="wave_qc_cause", **FLAG_ATTRS))
    ds["wave_qc_cause"] = ("time", cause, cause_attrs(WAVE_CAUSES, "parametri d'onda"))
    if "Dp" in w:
        ds["Dp"] = ("time", w.Dp.values.astype(np.float32), dict(
            standard_name="sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
            long_name="direzione di provenienza al picco, da Nord orario", units="degree",
            ancillary_variables="direction_qc"))
        ds["Dm"] = ("time", w.Dm.values.astype(np.float32), dict(
            standard_name="sea_surface_wave_from_direction",
            long_name="direzione media di provenienza, da Nord orario", units="degree",
            ancillary_variables="direction_qc"))
        dok = np.isfinite(w.Dp.values)
        dflag, dcause = combine_flags((n,), {"fuori_posizione": ~inpos, "movimento_telaio": mot & inpos,
                                             "bassa_energia": inpos & dok & (np.nan_to_num(hm) < hm0_dir_min),
                                             "nessun_dato": inpos & ~dok}, DIR_CAUSES)
        ds["direction_qc"] = ("time", dflag, dict(long_name="flag di qualita' delle direzioni",
                                                   ancillary_variables="direction_qc_cause",
                                                   comment=f"bassa_energia: Hm0 < {hm0_dir_min} m", **FLAG_ATTRS))
        ds["direction_qc_cause"] = ("time", dcause, cause_attrs(DIR_CAUSES, "direzione"))
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


def hourly_temperature(reader, deployment: dict | None, p_min: float = 1.0,
                       min_samples: int = 60) -> pd.DataFrame:
    """Statistiche orarie della temperatura da tutti i ping.

    Ore con strumento in posizione: statistiche dei soli ping in posizione, flag 1 (3 se
    meno di ``min_samples`` ping, cioè < 30 s di dati a 2 Hz). Ore interamente fuori
    posizione: statistiche di tutti i ping, flag 4. Ore senza dati: flag 9.
    """
    from .deployment import in_position
    ds = reader.load(["temperature", "pressure"])
    t = ds.time.values
    df = pd.DataFrame(dict(temp=ds.temperature.values.astype(float)), index=pd.DatetimeIndex(t))
    pos = np.ones(len(t), bool)
    if deployment is not None:
        pos &= in_position(t, deployment)
    if "pressure" in ds:
        pos &= ds.pressure.values >= p_min
    df["pos"] = pos
    agg = lambda x: x.resample("1h").agg(["mean", "std", "min", "max", "count"])
    a_in = agg(df.temp[df.pos]) if df.pos.any() else None
    a_all = agg(df.temp)
    out = a_all.copy()
    has_data = a_all["count"].values > 0
    has_in = np.zeros(len(out), bool)
    few = np.zeros(len(out), bool)
    if a_in is not None:
        a_in = a_in.reindex(a_all.index)
        has_in = (a_in["count"].fillna(0) > 0).values
        out.loc[has_in] = a_in.loc[has_in]
        few = has_in & (a_in["count"].fillna(0).values < min_samples)
    flag, cause = combine_flags((len(out),), {"fuori_posizione": has_data & ~has_in, "pochi_campioni": few,
                                              "nessun_dato": ~has_data}, TEMP_CAUSES)
    # minuti effettivamente campionati nell'ora (le misure sono solo durante i burst)
    mins = df.temp.resample("1min").count().resample("1h").apply(lambda x: int((x > 0).sum()))
    out["sampled_minutes"] = mins.reindex(out.index).fillna(0).astype(int)
    out["qc"] = flag.astype(np.int8)
    out["qc_cause"] = cause
    out = out.rename(columns={"mean": "temperature_mean", "std": "temperature_std", "min": "temperature_min",
                              "max": "temperature_max", "count": "n_samples"})
    out["n_samples"] = out["n_samples"].fillna(0).astype(int)
    out.index.name = "time_start"
    return out


def temperature_archive(h: pd.DataFrame, md: dict, src_attrs: dict, deployment: dict | None) -> xr.Dataset:
    t0 = h.index.values
    t1 = (h.index + pd.Timedelta(hours=1)).values
    ds = xr.Dataset(coords=dict(time=("time", t0 + (t1 - t0) / 2, dict(
        standard_name="time", long_name="centro dell'ora", axis="T", bounds="time_bnds"))))
    ds["time_bnds"] = (("time", "nv"), np.stack([t0, t1], axis=1))
    cm = "time: {} (interval: 1 hour)"
    ds["temperature"] = ("time", h.temperature_mean.values.astype(np.float32), dict(
        standard_name="sea_water_temperature", long_name="temperatura media oraria al fondo",
        units="degree_Celsius", cell_methods=cm.format("mean"), ancillary_variables="temperature_qc"))
    for k, ln, m in (("temperature_std", "deviazione standard nell'ora", "standard_deviation"),
                     ("temperature_min", "minimo nell'ora", "minimum"),
                     ("temperature_max", "massimo nell'ora", "maximum")):
        ds[k] = ("time", h[k].values.astype(np.float32), dict(
            long_name=f"temperatura: {ln}", units="degree_Celsius", cell_methods=cm.format(m)))
    ds["n_samples"] = ("time", h.n_samples.values.astype(np.int32), dict(long_name="ping usati", units="1"))
    ds["sampled_minutes"] = ("time", h.sampled_minutes.values.astype(np.int16), dict(
        long_name="minuti dell'ora con misure", units="min",
        comment="lo strumento misura solo durante i burst: la media e' sui minuti campionati"))
    ds["temperature_qc"] = ("time", h.qc.values, dict(
        long_name="flag di qualita' della temperatura oraria", ancillary_variables="temperature_qc_cause",
        **FLAG_ATTRS))
    ds["temperature_qc_cause"] = ("time", h.qc_cause.values.astype(np.uint16),
                                  cause_attrs(TEMP_CAUSES, "temperatura"))
    ds = _station_coords(ds, md)
    src = xr.Dataset(attrs=dict(src_attrs))
    if deployment:
        src.attrs.update(deployment_start=deployment["start"], deployment_end=deployment["end"],
                         deployment_method=deployment.get("method", ""))
    ds.attrs = _global_attrs(md, src, "temperature", h.index.values)
    ds.attrs["featureType"] = "timeSeries"
    ds.attrs["summary"] = ("Temperatura dell'acqua al fondo, media oraria di tutti i ping del sensore "
                           "dello strumento (Teledyne RDI Sentinel V)")
    return ds


def write_archive(outdir: str, md: dict, hab: float, verbose: bool = True, p_min: float = 1.0) -> list:
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
    c = wa = h = r = None
    bases = []
    if os.path.exists(pc):
        qc = xr.load_dataset(pc)
        c = currents_archive(qc, md, hab)
        base = os.path.join(adir, f"{sid}_{fmt(c.time.values.min())}_{fmt(c.time.values.max())}")
        bases.append(base)
        c.to_netcdf(base + "_currents.nc", encoding=_encoding(c))
        written.append(base + "_currents.nc")
        cols = [k for k in ("deployment_qc", "platform_motion", "temperature", "pressure", "water_depth",
                            "heading", "pitch", "roll", "n_pings", "in_position_frac",
                            "u_davg", "v_davg", "speed_davg", "dir_davg") if k in c]
        df = c[cols].to_dataframe()[cols]
        df.insert(1, "deployment_qc_cause", cause_text(c.deployment_qc_cause.values, POS_CAUSES))
        units = ", ".join(f"{k} [{c[k].attrs.get('units', '')}]" for k in cols)
        _write_csv(base + "_timeseries.csv", df, _csv_header(c, f"# colonne: {units}"))
        written.append(base + "_timeseries.csv")
        bits_doc = "# bit di causa: " + ", ".join(f"{b}={n}" for b, n, _ in VEL_CAUSES)
        for k in ("u", "v", "speed", "direction", "vel_qc", "vel_qc_cause"):
            m = pd.DataFrame(c[k].values, index=pd.DatetimeIndex(c.time.values, name="time"),
                             columns=[f"z{z:.2f}m" for z in c.z.values])
            extra = f"# {k} [{c[k].attrs.get('units', 'flag')}]; colonne = quota sul fondale"
            if k == "vel_qc_cause":
                extra += "\n" + bits_doc + " (valore = somma dei bit)"
            _write_csv(base + f"_currents_{k}.csv", m, _csv_header(c, extra))
            written.append(base + f"_currents_{k}.csv")
    if os.path.exists(pw):
        w = xr.load_dataset(pw)
        wa = waves_archive(w, md)
        base = os.path.join(adir, f"{sid}_{fmt(wa.time.values.min())}_{fmt(wa.time.values.max())}")
        bases.append(base)
        wa.to_netcdf(base + "_waves.nc", encoding=_encoding(wa))
        written.append(base + "_waves.nc")
        cols = [k for k in ("Hm0", "Tp", "Tm01", "Tm02", "Dp", "Dm", "wave_qc", "direction_qc",
                            "water_depth", "Hm0_pressure", "n_samples") if k in wa]
        df = wa[cols].to_dataframe()[cols]
        df.insert(df.columns.get_loc("wave_qc") + 1, "wave_qc_cause", cause_text(wa.wave_qc_cause.values, WAVE_CAUSES))
        if "direction_qc" in df:
            df.insert(df.columns.get_loc("direction_qc") + 1, "direction_qc_cause",
                      cause_text(wa.direction_qc_cause.values, DIR_CAUSES))
        units = ", ".join(f"{k} [{wa[k].attrs.get('units', 'flag')}]" for k in cols)
        _write_csv(base + "_waves.csv", df, _csv_header(wa, f"# metodo: {wa.attrs['wave_method']}\n# colonne: {units}"))
        written.append(base + "_waves.csv")
    # temperatura: media oraria da tutti i ping
    try:
        from .convert import load_inventory, StreamReader
        from .deployment import get_deployment
        inv = load_inventory(outdir)
        name = next((n for n, s_ in inv["streams"].items() if s_.get("label") in ("waves", "currents")), None)
    except FileNotFoundError:
        name = None
    if name:
        r = StreamReader(outdir, name)
        if "temperature" in r.variables():
            dep = get_deployment(outdir, r, p_min=p_min)
            h = hourly_temperature(r, dep, p_min=p_min)
            ta = temperature_archive(h, md, r.attrs, dep)
            base = os.path.join(adir, f"{sid}_{fmt(h.index.min())}_{fmt(h.index.max())}")
            enc = _encoding(ta)
            enc["time_bnds"] = dict(TIME_ENC)
            ta.to_netcdf(base + "_temperature_hourly.nc", encoding=enc)
            written.append(base + "_temperature_hourly.nc")
            cols = ["temperature_mean", "temperature_std", "temperature_min", "temperature_max",
                    "n_samples", "sampled_minutes", "qc"]
            hc = h[cols].copy()
            hc["qc_cause"] = cause_text(h.qc_cause.values, TEMP_CAUSES)
            _write_csv(base + "_temperature_hourly.csv", hc, _csv_header(
                ta, "# temperatura dell'acqua al fondo [degree_Celsius], media oraria; time_start = inizio "
                    "dell'ora; qc: 1 in posizione, 3 pochi dati, 4 fuori posizione, 9 nessun dato"))
            written.append(base + "_temperature_hourly.csv")
            try:
                _plot_temperature(h, os.path.join(outdir, "figures", "temperature_hourly.png"))
            except Exception as e:
                print(f"  (grafico temperatura non prodotto: {e})")
    if h is not None:
        try:
            from .plots import plot_temperature_overlays
            plot_temperature_overlays(h, wa, c, dep if name else None, os.path.join(outdir, "figures"))
        except Exception as e:
            print(f"  (grafici temperatura/onde/corrente non prodotti: {e})")
    if bases:
        from .report import write_report
        rep = write_report(outdir, md, hab, bases[0], c=c, wa=wa, h=h, written=list(written), reader=r)
        written.append(rep)
    if r is not None:
        r.close()
    guide = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LEGGIMI_DATI.md")
    if os.path.exists(guide):
        import shutil
        dst = os.path.join(adir, "LEGGIMI_DATI.md")
        shutil.copyfile(guide, dst)
        written.append(dst)
    if verbose:
        print("\nArchivio:")
        for f in written:
            print(f"  {f}")
    return written


def _plot_temperature(h, path):
    import matplotlib.pyplot as plt
    from . import plots as P
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.4), layout="constrained")
    good = h.qc.values == 1
    t = h.index.values
    ax.plot(t, np.where(good, h.temperature_mean, np.nan), color=P.C1, lw=1.2, label="in posizione")
    ax.plot(t[~good], h.temperature_mean.values[~good], "o", ms=2, color=P.AXIS,
            label="fuori posizione / pochi dati (flag 3-4)")
    ax.fill_between(t, np.where(good, h.temperature_min, np.nan), np.where(good, h.temperature_max, np.nan),
                    color=P.C1, alpha=0.15, lw=0)
    ax.set_ylabel("°C")
    if good.any():
        lo, hi = np.nanmin(h.temperature_min.values[good]), np.nanmax(h.temperature_max.values[good])
        pad = max(0.2, 0.05 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    nbad = int((h.qc.values != 1).sum())
    ax.set_title("Temperatura al fondo, media oraria (banda: minimo-massimo nell'ora)"
                 + (f"; {nbad} ore con flag 3/4/9, valori fuori scala non mostrati" if nbad else ""))
    ax.legend(loc="best", ncols=2)
    P._fmt_time(ax)
    P._save(fig, path)
