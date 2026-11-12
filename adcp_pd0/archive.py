"""Prodotti NetCDF (CF-1.8 + ACDD-1.3) e flag di qualità.

Livelli: L2 = dati puliti del pre-processing (correnti orarie, temperatura oraria, serie dei
burst d'onda); L3 = prodotti derivati del post-processing (parametri e spettri d'onda,
termoclino). ID del deployment = <stazione>_<inizio>_<fine> del periodo in posizione.

Flag di qualità (schema IOOS/QARTOD): 1 buono, 2 non valutato, 3 sospetto, 4 cattivo, 9 mancante;
accanto a ogni flag una variabile ``*_cause`` (somma di bit) dice il motivo.
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
              (8, "nessun_dato", 9), (16, "bassa_coerenza", 3)]
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
    "bassa_coerenza": "superficie e velocita' poco coerenti al picco dello spettro: direzione sospetta",
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


def code_version() -> str:
    """Versione del codice (commit git), se disponibile."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                           text=True, timeout=5).stdout.strip()
        b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=here, capture_output=True,
                           text=True, timeout=5).stdout.strip()
        return f"commit {h} (branch {b})" if h else "non disponibile"
    except Exception:
        return "non disponibile"


def deployment_id(md: dict, dep: dict | None = None, t=None) -> str:
    """ID unico del deployment: ``deployment_id`` dei metadati, altrimenti
    <stazione>_<inizio>_<fine> del periodo in posizione (o dei dati, se non determinato)."""
    if md.get("deployment_id"):
        return str(md["deployment_id"]).replace(" ", "_")
    sid = str(md.get("station_id") or "ADCP").replace(" ", "_")
    if dep:
        a, b = pd.Timestamp(dep["start"]), pd.Timestamp(dep["end"])
    else:
        a, b = pd.Timestamp(np.min(t)), pd.Timestamp(np.max(t))
    return f"{sid}_{a:%Y%m%d}_{b:%Y%m%d}"


KIND_INFO = {   # tipo di prodotto -> (featureType CF, descrizione)
    "currents": ("timeSeriesProfile", "Profili di corrente in coordinate Est/Nord/Su mediati nel tempo, con "
                 "flag di qualita' e causa; serie al fondo di temperatura, pressione e assetto dello strumento"),
    "temperature": ("timeSeries", "Temperatura dell'acqua al fondo: statistiche orarie di tutte le misure "
                    "del sensore dello strumento"),
    "wave_bursts": (None, "Serie pulite dei burst d'onda: elevazione della superficie dal beam verticale, "
                    "pressione, velocita' Est/Nord ad alcune quote; flag per burst"),
    "waves": ("timeSeries", "Parametri e spettri d'onda per burst: Hm0, periodi, direzione dai cross-spettri "
              "superficie-velocita', velocita' orbitale al fondo"),
    "thermocline": ("timeSeries", "Stima della quota del termoclino dal taglio verticale della corrente e "
                    "dal picco dell'eco (prodotto derivato, non validato con profili di temperatura)"),
}


def instrument_attrs(src: dict) -> dict:
    """Configurazione dello strumento (dal fixed leader) come attributi globali."""
    out = {}
    for k, name, scale in (("beam_angle_deg", "instrument_beam_angle_deg", 1), ("n_cells", "instrument_n_cells", 1),
                           ("cell_size_cm", "instrument_cell_size_m", 0.01),
                           ("bin1_dist_cm", "instrument_first_cell_range_m", 0.01),
                           ("blank_cm", "instrument_blank_m", 0.01),
                           ("heading_bias", "instrument_heading_bias_deg", 0.01),
                           ("pings_per_ens", "instrument_pings_per_ensemble", 1)):
        if src.get(k) is not None:
            try:
                out[name] = float(src[k]) * scale if scale != 1 else int(src[k])
            except (TypeError, ValueError):
                pass
    if src.get("orientation"):
        out["instrument_orientation"] = str(src["orientation"])
    cs = src.get("coord_system_recorded") or src.get("coord_system")
    if cs:
        out["instrument_coordinates_recorded"] = str(cs)
    return out


PROC_KEYS = ("hab", "p_atm", "nperseg", "fmin", "fmax", "kp2_min", "declination", "fs", "dt",
             "n_short_bursts_skipped", "hm0_dir_min", "coh_min", "direction_main_level", "wave_method",
             "has_vertical_beam")


def _global_attrs(md: dict, ds_src: xr.Dataset, kind: str, t: np.ndarray, level: str = "L2") -> dict:
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ft, summary = KIND_INFO.get(kind, (None, kind))
    t = np.asarray(t)
    a = {
        "Conventions": "CF-1.8, ACDD-1.3",
        "title": md.get("title") or f"ADCP Sentinel V, stazione {md.get('station_id') or 'ADCP'}: {kind}",
        "summary": summary,
        "product": kind,
        "data_level": level,
        "station_id": md.get("station_id") or "ADCP",
        "platform": "struttura sul fondale, strumento rivolto verso l'alto",
        "instrument": "Teledyne RDI Sentinel V",
        "instrument_serial_number": str(ds_src.attrs.get("serial", "")),
        "instrument_firmware": f"{ds_src.attrs.get('fw_ver', '')}.{ds_src.attrs.get('fw_rev', '')}",
        "time_coverage_start": str(pd.Timestamp(t.min())) if len(t) else "",
        "time_coverage_end": str(pd.Timestamp(t.max())) if len(t) else "",
        "date_created": now,
        "history": f"{now}: creato con adcp_pd0 (pre-processing, livello {level})",
        "software_version": code_version(),
        "processing_level": "dati controllati automaticamente (QC), non validati manualmente",
        "source_file": str(ds_src.attrs.get("source_file", "")),
    }
    if ft:
        a["featureType"] = ft
    a.update(instrument_attrs(ds_src.attrs))
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
        if k.startswith(("qc_", "avg_", "burst_")) or k in PROC_KEYS:
            a[f"proc_{k}"] = v if not isinstance(v, (bool, np.bool_)) else int(v)
    return a


L3_KEEP = ("Conventions", "title", "station_id", "platform", "instrument", "instrument_serial_number",
           "instrument_firmware", "source_file", "institution", "project", "creator_name", "creator_email",
           "license", "deployment_id", "deployment_start", "deployment_end", "deployment_method", "time_zone",
           "comment", "geospatial_vertical_reference", "magnetic_declination_deg",
           "magnetic_declination_applied", "geospatial_lat_min", "geospatial_lat_max", "geospatial_lon_min",
           "geospatial_lon_max", "instrument_height_above_bed_m", "processing_level")


def l3_attrs(src_attrs: dict, kind: str, t, proc: dict | None = None) -> dict:
    """Attributi globali di un prodotto derivato (L3) a partire da quelli del file L2 di origine."""
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ft, summary = KIND_INFO.get(kind, (None, kind))
    t = np.asarray(t)
    a = {k: src_attrs[k] for k in src_attrs if k in L3_KEEP or k.startswith("instrument_")}
    a.update(summary=summary, product=kind, data_level="L3", date_created=now,
             history=f"{now}: creato con adcp_pd0 (post-processing, livello L3)",
             software_version=code_version(),
             time_coverage_start=str(pd.Timestamp(t.min())) if len(t) else "",
             time_coverage_end=str(pd.Timestamp(t.max())) if len(t) else "")
    if ft:
        a["featureType"] = ft
    for k, v in (proc or {}).items():
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
    if gp is not None:
        ds["valid_ping_fraction"] = (dims, gp.astype(np.float32), dict(
            long_name="frazione di ping validi (correlazione sopra la soglia) nell'intervallo", units="1"))
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


def wave_bursts_archive(b: xr.Dataset, md: dict, dep: dict | None = None, p_min: float = 1.0) -> xr.Dataset:
    """Prodotto L2 delle serie dei burst d'onda: flag per burst e metadati CF."""
    from .deployment import in_position, motion_overlap
    ds = b.copy()
    nb = ds.sizes["time"]
    dt = float(b.attrs.get("dt", 0.5))
    t0 = ds.time.values
    t1 = t0 + (ds["n_samples"].values.astype(float) * dt * 1e6).astype("timedelta64[us]")
    inpos, mot = np.ones(nb, bool), np.zeros(nb, bool)
    if dep is not None:
        inpos = in_position(t0, dep) & in_position(t1 - np.timedelta64(1, "us"), dep)
        mot = motion_overlap(t0, t1, dep)
    inpos &= np.nan_to_num(ds["depth_sensor"].values) * 1025.0 * 9.81 / 1e4 >= p_min
    has_eta = "eta" in ds
    data_ok = np.isfinite(ds["eta"].values).any(axis=1) if has_eta else np.isfinite(ds["pressure"].values).any(axis=1)
    sf = np.nan_to_num(ds["surface_found_frac"].values, nan=0.0) if has_eta else np.ones(nb)
    flag, cause = combine_flags((nb,), {
        "fuori_posizione": ~inpos, "movimento_telaio": mot & inpos,
        "superficie_trovata_parzialmente": inpos & data_ok & (sf < 0.9),
        "nessun_dato": inpos & ~data_ok}, WAVE_CAUSES)
    ds["burst_qc"] = ("time", flag, dict(long_name="flag di qualita' del burst",
                                         ancillary_variables="burst_qc_cause", **FLAG_ATTRS))
    ds["burst_qc_cause"] = ("time", cause, cause_attrs(WAVE_CAUSES, "burst"))
    ds["in_position"] = ("time", inpos.astype(np.int8), dict(long_name="1 = strumento in posizione sul fondo"))
    ds["platform_motion"] = ("time", mot.astype(np.int8), dict(long_name="1 = telaio in movimento durante il burst"))
    hab = float(b.attrs.get("hab", md.get("instrument_height_above_bed_m") or 0.6))
    va = dict(
        pressure=dict(standard_name="sea_water_pressure", units="dbar", long_name="pressione al sensore"),
        surface_range=dict(units="m", long_name="distanza trasduttore-superficie dal massimo di eco del beam "
                           "verticale (grezza; mancante dove il picco non e' netto)"),
        eta=dict(units="m", long_name="elevazione della superficie rispetto alla media del burst (pulita)",
                 comment="spike oltre 4 MAD rimossi, buchi interpolati se < 15% dei ping; mancante per "
                         "tutto il burst se i buchi sono di piu'"),
        u=dict(standard_name="eastward_sea_water_velocity", units="m s-1", long_name="velocita' verso Est"),
        v=dict(standard_name="northward_sea_water_velocity", units="m s-1", long_name="velocita' verso Nord"),
        corr_ok_frac=dict(units="1", long_name="frazione di ping con correlazione sopra la soglia"),
        n_samples=dict(units="1", long_name="ping nel burst"),
        depth_sensor=dict(units="m", long_name="profondita' del sensore (pressione media / (rho g))"),
        water_depth=dict(standard_name="sea_floor_depth_below_sea_surface", units="m",
                         long_name="profondita' totale media nel burst"),
        surface_found_frac=dict(units="1", long_name="frazione di ping con superficie trovata"),
        surface_range_mean=dict(units="m", long_name="distanza media trasduttore-superficie nel burst"),
        heading=dict(units="degree", long_name="heading medio nel burst"),
        pitch=dict(units="degree", long_name="pitch medio nel burst"),
        roll=dict(units="degree", long_name="roll medio nel burst"))
    for k, at in va.items():
        if k in ds:
            ds[k].attrs.update(at)
    ds["time"].attrs.update(standard_name="time", long_name="inizio del burst", axis="T")
    ds["sample_time"].attrs.update(units="s", long_name="tempo dall'inizio del burst")
    if "level_range" in ds.coords:
        ds["level_range"].attrs.update(units="m", long_name="distanza della cella dal trasduttore")
        ds = ds.assign_coords(level_height=("level", (ds["level_range"].values + hab).astype(np.float32),
                                            dict(units="m", standard_name="height_above_sea_floor",
                                                 long_name="quota della cella sul fondale")))
        ds["level_cell"].attrs.update(long_name="indice (da 0) della cella nello stream")
    ds = _station_coords(ds, md)
    src = xr.Dataset(attrs=dict(b.attrs))
    if dep:
        src.attrs.update(deployment_start=dep["start"], deployment_end=dep["end"],
                         deployment_method=dep.get("method", ""))
    ds.attrs = _global_attrs(md, src, "wave_bursts", t0)
    return ds


WAVE_VAR_ATTRS = dict(
    Hm0=dict(standard_name="sea_surface_wave_significant_height", units="m",
             long_name="altezza significativa spettrale Hm0 = 4 sqrt(m0)", ancillary_variables="wave_qc"),
    Tp=dict(standard_name="sea_surface_wave_period_at_variance_spectral_density_maximum", units="s",
            long_name="periodo di picco", ancillary_variables="wave_qc"),
    Tm01=dict(standard_name="sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
              units="s", long_name="periodo medio Tm01", ancillary_variables="wave_qc"),
    Tm02=dict(standard_name="sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment",
              units="s", long_name="periodo medio Tm02", ancillary_variables="wave_qc"),
    Ub=dict(units="m s-1", long_name="ampiezza significativa della velocita' orbitale al fondo (teoria lineare, "
            "2 volte la deviazione standard)", ancillary_variables="wave_qc"),
    S=dict(standard_name="sea_surface_wave_variance_spectral_density", units="m2 Hz-1",
           long_name="spettro di elevazione"),
    Hm0_pressure=dict(units="m", long_name="Hm0 da pressione (solo frequenze sotto la frequenza di taglio)",
                      comment="in acqua profonda sottostima: solo per confronto"),
    fcut_pressure=dict(units="Hz", long_name="frequenza di taglio per la pressione"),
    S_pressure=dict(units="m2 Hz-1", long_name="spettro di elevazione da pressione (fino alla frequenza di taglio)"),
    Dp=dict(standard_name="sea_surface_wave_from_direction_at_variance_spectral_density_maximum", units="degree",
            long_name="direzione di provenienza al picco, da Nord orario (quota principale)",
            ancillary_variables="direction_qc"),
    Dm=dict(standard_name="sea_surface_wave_from_direction", units="degree",
            long_name="direzione media di provenienza, da Nord orario (quota principale)",
            ancillary_variables="direction_qc"),
    Dp_level=dict(units="degree", long_name="direzione di provenienza al picco stimata a ogni quota"),
    Dm_level=dict(units="degree", long_name="direzione media di provenienza stimata a ogni quota"),
    dir_level_spread=dict(units="degree", long_name="massima differenza di Dp fra le quote (indice di incertezza)"),
    dir_coherence=dict(units="1", long_name="coerenza^2 fra elevazione e velocita' orizzontali al picco"),
    direction_f=dict(units="degree", long_name="direzione di provenienza per frequenza (quota principale)"),
    water_depth=dict(standard_name="sea_floor_depth_below_sea_surface", units="m",
                     long_name="profondita' totale media nel burst"),
    n_samples=dict(units="1", long_name="ping nel burst"),
    surface_found_frac=dict(units="1", long_name="frazione di ping con superficie trovata"),
    in_position=dict(long_name="1 = strumento in posizione sul fondo"),
    platform_motion=dict(long_name="1 = telaio in movimento durante il burst"),
)


def waves_l3(w: xr.Dataset, src_attrs: dict) -> xr.Dataset:
    """Prodotto L3 delle onde: attributi CF delle variabili, dei flag e globali."""
    ds = w.copy()
    for k, at in WAVE_VAR_ATTRS.items():
        if k in ds:
            ds[k].attrs.update(at)
    ds["time"].attrs.update(standard_name="time", long_name="inizio del burst", axis="T")
    ds["frequency"].attrs.update(standard_name="sea_surface_wave_frequency", units="Hz")
    if "level_range" in ds.coords:
        ds["level_range"].attrs.update(units="m", long_name="distanza della cella dal trasduttore")
    ds["wave_qc"].attrs.update(long_name="flag di qualita' dei parametri d'onda",
                               ancillary_variables="wave_qc_cause", **FLAG_ATTRS)
    ds["wave_qc_cause"].attrs.update(cause_attrs(WAVE_CAUSES, "parametri d'onda"))
    if "direction_qc" in ds:
        ds["direction_qc"].attrs.update(long_name="flag di qualita' delle direzioni",
                                        ancillary_variables="direction_qc_cause", **FLAG_ATTRS)
        ds["direction_qc_cause"].attrs.update(cause_attrs(DIR_CAUSES, "direzione"))
    ds.attrs = l3_attrs(src_attrs, "waves", ds.time.values, dict(w.attrs))
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
