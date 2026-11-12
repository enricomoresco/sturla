"""Post-processing: analisi e grafici dai dati puliti (L2) del pre-processing.

  python postprocessing.py -o out

Legge solo i NetCDF di out/preprocessing/ (mai i ping grezzi) e scrive in out/postprocessing/:
  <ID>_L3_waves.nc         parametri e spettri d'onda per burst
  <ID>_L3_thermocline.nc   stima della quota del termoclino
  tabelle/*.csv            serie orarie, onde, mareggiate, statistiche mensili, ...
  figure/*.png             correnti, onde, temperatura, termoclino, assetto, livello, qualita'
  risultati.json           tutti i numeri chiave
  <ID>_SINTESI_RISULTATI.md
  <ID>_risultati_per_report.zip   figure + tabelle + numeri + file piccoli, da inviare per il report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

from . import postplots as PP
from . import stats as ST
from . import thermocline as TH
from .archive import (DIR_CAUSES, WAVE_CAUSES, _encoding, cause_text, code_version, waves_l3)
from .waves import wave_spectra

RHO, G = 1025.0, 9.81


def _find(pre, suffix):
    ff = sorted(glob.glob(os.path.join(pre, f"*{suffix}")))
    return ff[-1] if ff else None


def _clean(o):
    """Rende un oggetto scrivibile in JSON (NaN -> null, tipi numpy/pandas -> Python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _clean(o.reset_index().to_dict("records"))
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(o) else str(pd.Timestamp(o))[:19]
    if isinstance(o, (pd.Timedelta, np.timedelta64)):
        return str(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6) if np.isfinite(o) else None
    if o is pd.NaT:
        return None
    return o


def _csv(path, df, index_label=None):
    df.to_csv(path, encoding="utf-8", date_format="%Y-%m-%dT%H:%M:%S", float_format="%.4f",
              index_label=index_label)
    return path


def write_summary(path, res) -> str:
    """Sintesi leggibile dei risultati (Markdown)."""
    g = lambda d, *ks, default="n.d.": _get(d, ks, default)
    L = [f"# Sintesi dei risultati - {res.get('id', '')}", "",
         f"Generata il {res.get('creato', '')} da `adcp_pd0` ({res.get('versione_codice', '')}). "
         "Numeri completi in `risultati.json`, tabelle in `tabelle/`, grafici in `figure/`.", ""]
    d = res.get("deployment", {})
    L += ["## Periodo", "", f"- In posizione: {g(d, 'inizio')} -> {g(d, 'fine')} ({g(d, 'giorni_in_posizione')} giorni)",
          f"- Dati registrati: {g(d, 'dati_da')} -> {g(d, 'dati_a')}",
          f"- Movimenti del telaio: {len(res.get('movimenti_telaio', []))}", ""]
    o = res.get("onde", {})
    if o.get("n_validi"):
        L += ["## Onde", "", f"- Burst validi: {o['n_validi']} su {o['n_burst']}",
              f"- Hm0: media {g(o, 'Hm0', 'media')} m, mediana {g(o, 'Hm0', 'mediana')} m, 95° percentile "
              f"{g(o, 'Hm0', 'p95')} m, massimo {g(o, 'Hm0', 'max')} m il {g(o, 'Hm0', 'quando_max')}",
              f"- Tp mediano {g(o, 'Tp', 'mediana')} s; Tm02 mediano {g(o, 'Tm02', 'mediana')} s",
              f"- Direzione prevalente (Hm0 >= {g(o, 'direzione', 'soglia_Hm0')} m): "
              f"{g(o, 'direzione', 'settore_prevalente')}; pesata sull'energia {g(o, 'direzione', 'media_pesata_energia')}°",
              f"- Mareggiate (Hm0 >= {g(res, 'parametri', 'soglia_mareggiata_m')} m): {len(res.get('mareggiate', []))}", ""]
        st = res.get("mareggiate", [])
        if st:
            L += ["| inizio | fine | durata (h) | Hm0 max (m) | Tp (s) | Dp (°) | Ub max (m/s) | telaio mosso |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in st:
                L.append(f"| {r.get('inizio')} | {r.get('fine')} | {_n(r.get('durata_h'), 0)} | {_n(r.get('Hm0_max'))} | "
                         f"{_n(r.get('Tp'), 1)} | {_n(r.get('Dp'), 0)} | {_n(r.get('Ub_max'))} | "
                         f"{'si' if r.get('movimento_telaio') else 'no'} |")
            L.append("")
    cr = res.get("correnti", {})
    if cr.get("strati"):
        L += ["## Correnti", "", "| strato | quota | media (m/s) | 95° perc. | massimo | vettore medio | settore prevalente |",
              "|---|---|---|---|---|---|---|"]
        for k, v in cr["strati"].items():
            L.append(f"| {k} | {v.get('quota', '')} | {_n(v.get('media'), 3)} | {_n(v.get('p95'), 3)} | {_n(v.get('max'), 3)} | "
                     f"{_n(v.get('vettore_medio_vel'), 3)} m/s verso {_n(v.get('vettore_medio_dir'), 0)}° | "
                     f"{v.get('settore_prevalente', '')} |")
        pa = cr.get("asse_principale", {})
        L += ["", f"Asse principale della corrente media: {g(pa, 'direzione_gradi')}° "
              f"({_n(100 * (pa.get('frazione_varianza') or np.nan), 0)}% della varianza).", ""]
    t = res.get("temperatura", {})
    if t.get("n_ore"):
        mv = t.get("massima_variazione", {})
        L += ["## Temperatura al fondo", "", f"- Media {g(t, 'media')} °C; minimo {g(t, 'min')} °C il {g(t, 'quando_min')}; "
              f"massimo {g(t, 'max')} °C il {g(t, 'quando_max')}",
              f"- Massima variazione in 4 giorni: {g(mv, 'delta')} °C fra {g(mv, 'da')} e {g(mv, 'a')}",
              f"- Periodi dominanti nel periodo di massima variazione: "
              + ", ".join(f"{p['periodo_h']} h" for p in t.get("periodi_dominanti_zoom", [])), ""]
    tc = res.get("termoclino", {})
    if tc.get("n_intervalli"):
        L += ["## Termoclino (stima indiretta)", "",
              f"- Stime affidabili dal taglio: {g(tc, 'affidabili_pct')}% degli intervalli",
              "- Quota mediana mensile dal taglio (m dal fondale): "
              + ", ".join(f"{k}: {v}" for k, v in tc.get("taglio_mediana_mensile", {}).items()),
              "- Quota mediana mensile dall'eco: "
              + ", ".join(f"{k}: {v}" for k, v in tc.get("eco_mediana_mensile", {}).items()), ""]
    m = res.get("marea", {})
    if m.get("costituenti"):
        L += ["## Livello e marea", "", "- " + "; ".join(f"{k}: {v['ampiezza_cm']} cm" for k, v in m["costituenti"].items()),
              f"- Varianza spiegata dalla marea: {_n(100 * (m.get('varianza_spiegata') or np.nan), 0)}%; "
              f"residuo {g(m, 'residuo_std_cm')} cm (dev. std)", ""]
    mt = res.get("movimenti_telaio", [])
    if mt:
        L += ["## Movimenti del telaio", "", "| inizio | fine | rotazione heading (°) | Hm0 max nelle 3 h prima (m) | Ub max (m/s) |",
              "|---|---|---|---|---|"]
        for r in mt:
            L.append(f"| {r.get('inizio')} | {r.get('fine')} | {_n(r.get('rotazione_heading'), 1)} | "
                     f"{_n(r.get('Hm0_max_prima'))} | {_n(r.get('Ub_max_prima'))} |")
        L.append("")
    if res.get("errori"):
        L += ["## Passi con errori", ""] + [f"- {e}" for e in res["errori"]] + [""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


def _get(d, ks, default="n.d."):
    for k in ks:
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def _n(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n.d."
    return f"{x:.{nd}f}" if np.isfinite(x) else "n.d."


def run(a) -> dict:
    out = a.out
    pre, post = os.path.join(out, "preprocessing"), os.path.join(out, "postprocessing")
    fdir, tdir = os.path.join(post, "figure"), os.path.join(post, "tabelle")
    fc, ft, fb = (_find(pre, s) for s in ("_L2_currents.nc", "_L2_temperature.nc", "_L2_wave_bursts.nc"))
    first = fc or ft or fb
    if first is None:
        raise SystemExit(f"nessun file L2 in {pre}: lanciare prima  python preprocessing.py <file.pd0> -o {out}")
    did = os.path.basename(first).split("_L2_")[0]
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    dep = None
    if os.path.exists(os.path.join(pre, "deployment.json")):
        with open(os.path.join(pre, "deployment.json"), encoding="utf-8") as fh:
            dep = json.load(fh)
    errors, figures, tables, products = [], [], [], []

    def step(name, fn, default=None):
        try:
            return fn()
        except Exception:
            tb = traceback.format_exc()
            errors.append((name, tb))
            print(f"  ERRORE in '{name}': {tb.strip().splitlines()[-1]} (il resto prosegue)")
            return default

    def figure(name, fn):
        p = step(f"figura {name}", lambda: fn(os.path.join(fdir, name)))
        if p:
            figures.append(p)

    print(f"Post-processing di {did}")
    c = xr.load_dataset(fc) if fc else None
    tds = xr.load_dataset(ft) if ft else None
    b = xr.load_dataset(fb) if fb else None
    src = (c if c is not None else (tds if tds is not None else b)).attrs
    lat = src.get("geospatial_lat_min")
    res = dict(id=did, creato=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
               versione_codice=code_version(),
               parametri=dict(soglia_mareggiata_m=a.soglia_mareggiata, nperseg=a.nperseg, fmin=a.fmin, fmax=a.fmax,
                              kp2_min=a.kp2_min, hm0_dir_min=a.hm0_dir_min))

    # ---- serie di base
    Th = T = level = layers = None
    if tds is not None:
        Th = pd.DataFrame(dict(T=tds["temperature"].values.astype(float),
                               Tmin=tds["temperature_min"].values.astype(float),
                               Tmax=tds["temperature_max"].values.astype(float),
                               qc=tds["temperature_qc"].values), index=pd.DatetimeIndex(tds["time"].values))
        T = Th["T"].where(Th["qc"].isin([1, 3]))
    if c is not None:
        inpos = c["deployment_qc"].values == 1
        tc_ = pd.DatetimeIndex(c["time"].values)
        if Th is None and "temperature" in c:
            Th = pd.DataFrame(dict(T=c["temperature"].values.astype(float), Tmin=np.nan, Tmax=np.nan,
                                   qc=c["deployment_qc"].values), index=tc_)
            T = Th["T"].where(Th["qc"].isin([1, 3]))
        if "pressure" in c and inpos.any():
            lv = np.where(inpos, c["pressure"].values.astype(float) * 1e4 / (RHO * G), np.nan)
            level = pd.Series(lv - np.nanmean(lv), index=tc_)
        layers = step("strati di corrente", lambda: ST.layer_series(c))

    # ---- onde (L3)
    w = None
    if b is not None:
        print("Onde: spettri, parametri e direzione")

        def _waves():
            ws = wave_spectra(b, nperseg=a.nperseg, fmin=a.fmin, fmax=a.fmax, kp2_min=a.kp2_min,
                              hm0_dir_min=a.hm0_dir_min)
            wl = waves_l3(ws, dict(b.attrs))
            p = os.path.join(post, f"{did}_L3_waves.nc")
            wl.to_netcdf(p, encoding=_encoding(wl))
            products.append(p)
            return wl
        w = step("onde", _waves)

    # ---- termoclino (L3)
    th = th_extra = None
    if c is not None:
        print("Termoclino: stima dal taglio della corrente e dall'eco")

        def _thermo():
            df, ex = TH.thermocline(c)
            if df.empty:
                return None, None
            ds = TH.to_dataset(df, dict(c.attrs), ex["params"])
            p = os.path.join(post, f"{did}_L3_thermocline.nc")
            ds.to_netcdf(p, encoding=_encoding(ds))
            products.append(p)
            return df, ex
        th, th_extra = step("termoclino", _thermo, default=(None, None))

    # ---- statistiche
    print("Statistiche ed eventi")
    if dep:
        t0, t1 = pd.Timestamp(dep["start"]), pd.Timestamp(dep["end"])
        res["deployment"] = dict(inizio=str(t0)[:19], fine=str(t1)[:19], metodo=dep.get("method"),
                                 dati_da=str(dep.get("data_start", ""))[:19], dati_a=str(dep.get("data_end", ""))[:19],
                                 giorni_in_posizione=round((t1 - t0).total_seconds() / 86400.0, 1),
                                 pressione_riferimento_dbar=dep.get("p_ref_dbar"))
    res["strumento"] = {k: _clean(v) for k, v in src.items()
                        if k.startswith("instrument") or k in ("station_id", "geospatial_lat_min", "geospatial_lon_min",
                                                               "time_zone", "source_file")}
    spec = bands = pa = None
    if layers is not None:
        def _cur():
            out_ = dict(strati={}, definizione_strati={k: layers.attrs.get(k) for k in ("fondo", "meta", "sup", "media")},
                        profondita_mediana_m=layers.attrs.get("profondita_mediana"))
            for k, nme in (("sup", "superficie"), ("meta", "meta_colonna"), ("fondo", "fondo"), ("media", "media_verticale")):
                s = ST.speed_stats(layers[f"u_{k}"].values, layers[f"v_{k}"].values, layers.index)
                s["quota"] = layers.attrs.get(k)
                out_["strati"][nme] = s
            out_["asse_principale"] = ST.principal_axis(layers["u_media"].values, layers["v_media"].values)
            return out_
        res["correnti"] = step("statistiche correnti", _cur, default={})
        pa = res["correnti"].get("asse_principale")
        dt_h = float(np.median(np.diff(layers.index.values).astype("timedelta64[s]").astype(float))) / 3600.0 \
            if len(layers) > 2 else 1.0

        def _spec():
            for k, nme in (("sup", "superficie"), ("media", "media sulla verticale")):
                sp = ST.rotary_spectrum(layers[f"u_{k}"].values, layers[f"v_{k}"].values, dt_hours=dt_h)
                if sp:
                    return sp, ST.band_variance(sp, lat), nme
            return None, None, ""
        spec, bands, spec_label = step("spettro correnti", _spec, default=(None, None, ""))
        if bands:
            res["correnti"]["spettro"] = dict(strato=spec_label, ore_usate=spec["n_ore"], **bands)
    if w is not None:
        res["onde"] = step("statistiche onde", lambda: ST.wave_stats(w), default={})
    st = None
    if w is not None:
        st = step("mareggiate", lambda: ST.enrich_storms(ST.storms(w, a.soglia_mareggiata), layers, T, dep))
        if st is not None:
            res["mareggiate"] = st
    mt = step("movimenti telaio", lambda: ST.motion_table(dep, w)) if dep else None
    res["movimenti_telaio"] = mt if mt is not None else []
    zoom = None
    if T is not None:
        res["temperatura"] = step("statistiche temperatura", lambda: ST.temperature_stats(T), default={})
        zoom = step("periodo di massima variazione", lambda: ST.max_change_window(T.resample("1D").mean()))
        if zoom:
            res["temperatura"]["periodi_dominanti_zoom"] = step(
                "periodi dominanti", lambda: ST.dominant_periods(T[zoom["t0"]:zoom["t1"]]), default=[])
            res["temperatura"]["finestra_zoom"] = dict(da=zoom["t0"], a=zoom["t1"])
    if th is not None:
        res["termoclino"] = step("sintesi termoclino", lambda: TH.summary(th), default={})
        res["termoclino"]["parametri"] = th_extra["params"]
    tide = None
    if level is not None:
        tide = step("marea", lambda: ST.tide_fit(level.index, level.values))
        if tide:
            res["marea"] = tide["info"]
    monthly = step("tabella mensile", lambda: ST.monthly_table(w, layers, T, th))
    if monthly is not None and len(monthly):
        res["mensile"] = monthly
    qc = step("qualita'", lambda: ST.qc_summary(c, tds, b, w), default={})
    res["qualita"] = qc

    # ---- tabelle
    def _tables():
        if w is not None:
            cols = [k for k in ("Hm0", "Tp", "Tm01", "Tm02", "Dp", "Dm", "Ub", "wave_qc", "direction_qc",
                                "dir_coherence", "dir_level_spread", "Hm0_pressure", "fcut_pressure", "water_depth",
                                "n_samples", "surface_found_frac") if k in w]
            df = w[cols].to_dataframe()[cols]
            df.insert(df.columns.get_loc("wave_qc") + 1, "wave_qc_cause", cause_text(w["wave_qc_cause"].values, WAVE_CAUSES))
            if "direction_qc" in df:
                df.insert(df.columns.get_loc("direction_qc") + 1, "direction_qc_cause",
                          cause_text(w["direction_qc_cause"].values, DIR_CAUSES))
            tables.append(_csv(os.path.join(tdir, "onde_parametri.csv"), df, "inizio_burst"))
        if st is not None and len(st):
            tables.append(_csv(os.path.join(tdir, "mareggiate.csv"), st.set_index("inizio")))
        if mt is not None and len(mt):
            tables.append(_csv(os.path.join(tdir, "movimenti_telaio.csv"), mt.set_index("inizio")))
        if monthly is not None and len(monthly):
            tables.append(_csv(os.path.join(tdir, "statistiche_mensili.csv"), monthly))
        if th is not None:
            tables.append(_csv(os.path.join(tdir, "termoclino.csv"), th))
        # distribuzioni congiunte direzione-intensita' (percentuali dei casi)
        if w is not None and "Dp" in w:
            gd = ST.dir_good(w) & ST.wave_good(w)
            if gd.any():
                tables.append(_csv(os.path.join(tdir, "onde_distribuzione_Hm0_direzione.csv"), ST.joint_table(
                    w["Dp"].values[gd], w["Hm0"].values[gd], [0, 0.5, 1.0, 1.5, np.inf], "m")))
        if layers is not None:
            for k, nme in (("sup", "superficie"), ("meta", "meta_colonna"), ("fondo", "fondo")):
                if layers[f"vel_{k}"].notna().any():
                    tables.append(_csv(os.path.join(tdir, f"correnti_distribuzione_{nme}.csv"), ST.joint_table(
                        layers[f"dir_{k}"].values, layers[f"vel_{k}"].values, [0, 0.05, 0.10, 0.20, np.inf], "m/s")))
        hourly = {}
        if Th is not None:
            hourly.update(temperatura=Th["T"], temperatura_min=Th["Tmin"], temperatura_max=Th["Tmax"],
                          temperatura_qc=Th["qc"])
        if c is not None:
            tc2 = pd.DatetimeIndex(c["time"].values)
            for k in ("pressure", "water_depth", "heading", "pitch", "roll", "deployment_qc", "platform_motion", "n_pings"):
                if k in c:
                    hourly[k] = pd.Series(c[k].values, index=tc2)
        if layers is not None:
            for k in ("sup", "meta", "fondo", "media"):
                hourly[f"corrente_{k}_vel"] = layers[f"vel_{k}"]
                hourly[f"corrente_{k}_dir"] = layers[f"dir_{k}"]
        if tide:
            hourly.update(livello_m=tide["livello"], marea_m=tide["marea"], residuo_m=tide["residuo"])
        if hourly:
            tables.append(_csv(os.path.join(tdir, "serie_orarie.csv"), pd.DataFrame(hourly).sort_index(), "tempo"))
        if tide:
            tables.append(_csv(os.path.join(tdir, "marea_costituenti.csv"),
                               pd.DataFrame(tide["info"]["costituenti"]).T, "costituente"))
    step("tabelle", _tables)

    # ---- figure
    print("Figure")
    wd_med = float(np.nanmedian(c["water_depth"].values[c["deployment_qc"].values == 1])) \
        if c is not None and "water_depth" in c and (c["deployment_qc"].values == 1).any() else None
    figure("01_quadro_generale.png", lambda p: PP.fig_overview(p, did, w, layers, T, level, dep))
    if c is not None:
        figure("02_assetto_strumento.png", lambda p: PP.fig_attitude(p, c, dep, w))
    figure("03_qualita_dati.png", lambda p: PP.fig_quality(p, c, tds, w, qc))
    if c is not None:
        figure("10_correnti_sezioni.png", lambda p: PP.fig_current_sections(p, c))
        figure("11_correnti_profilo_medio.png", lambda p: PP.fig_current_profile(p, c))
        figure("15_eco_sezione.png", lambda p: PP.fig_echo_section(p, c, th))
    if layers is not None:
        figure("12_correnti_rose.png", lambda p: PP.fig_current_roses(p, layers))
        figure("13_correnti_strati.png", lambda p: PP.fig_current_layers(p, layers, pa, dep))
        figure("14_correnti_spettro.png", lambda p: PP.fig_current_spectrum(p, spec, bands, spec_label if spec else ""))
        figure("16_correnti_vettore_progressivo.png", lambda p: PP.fig_current_pvd(p, layers))
    if w is not None:
        figure("20_onde_serie.png", lambda p: PP.fig_wave_series(p, w, dep))
        figure("21_onde_spettrogramma.png", lambda p: PP.fig_wave_spectrogram(p, w, a.fmax))
        figure("22_onde_spettri_medi.png", lambda p: PP.fig_wave_mean_spectra(p, w, st, a.fmax))
        figure("23_onde_rosa.png", lambda p: PP.fig_wave_rose(p, w))
        figure("24_onde_hm0_tp.png", lambda p: PP.fig_wave_hm0_tp(p, w))
        figure("25_onde_direzione_frequenza.png", lambda p: PP.fig_wave_dir_freq(p, w))
        figure("26_onde_mareggiate.png", lambda p: PP.fig_wave_storms(p, w, st, dep))
        figure("27_onde_confronto_pressione.png", lambda p: PP.fig_wave_pressure(p, w))
        figure("28_onde_distribuzione.png", lambda p: PP.fig_wave_distribution(p, w))
    if Th is not None:
        figure("30_temperatura.png", lambda p: PP.fig_temperature(p, Th))
        if w is not None:
            hm = pd.Series(np.where(ST.wave_good(w), w["Hm0"].values, np.nan), index=pd.DatetimeIndex(w["time"].values))
            figure("31_temperatura_e_Hm0.png", lambda p: PP.fig_temp_overlay(
                p, Th, hm, "Hm0", "Altezza d'onda significativa Hm0 (m)", PP.C2, dep,
                "Temperatura al fondo e altezza d'onda significativa",
                "Temperatura: media giornaliera delle medie orarie (giorni misti esclusi). Hm0: un valore per burst.",
                fine=True))
        if layers is not None:
            for k, nme, fn in (("fondo", "al fondo", "32_temperatura_e_corrente_fondo.png"),
                               ("sup", "in superficie", "33_temperatura_e_corrente_superficie.png")):
                figure(fn, lambda p, k=k, nme=nme: PP.fig_temp_overlay(
                    p, Th, layers[f"vel_{k}"], f"Corrente {nme}", f"Intensità della corrente {nme} (m/s)", PP.C3, dep,
                    f"Temperatura al fondo e corrente {nme}",
                    f"Corrente {nme}: {layers.attrs.get(k, '')}, solo celle con flag 1-3.", fine=False))
        figure("34_temperatura_zoom.png", lambda p: PP.fig_temp_zoom(p, Th, zoom, level, w, layers, dep))
        figure("35_temperatura_periodi.png", lambda p: PP.fig_temp_periods(p, Th, zoom, lat))
    if th is not None:
        figure("40_termoclino.png", lambda p: PP.fig_thermocline(p, th, w, Th, dep, th_extra["params"]))
        figure("41_termoclino_taglio.png", lambda p: PP.fig_thermocline_shear(p, th, th_extra, wd_med))
    figure("50_livello_mare.png", lambda p: PP.fig_sea_level(p, tide))

    # ---- risultati, sintesi, zip
    res["figure"] = [os.path.basename(f) for f in figures]
    res["tabelle"] = [os.path.basename(f) for f in tables]
    res["prodotti"] = [os.path.basename(f) for f in products]
    res["errori"] = [f"{n}: {tb.strip().splitlines()[-1]}" for n, tb in errors]
    rj = os.path.join(post, "risultati.json")
    with open(rj, "w", encoding="utf-8") as fh:
        json.dump(_clean(res), fh, indent=2, ensure_ascii=False)
    sm = step("sintesi", lambda: write_summary(os.path.join(post, f"{did}_SINTESI_RISULTATI.md"), _clean(res)))
    err_path = os.path.join(post, "errori_postprocessing.txt")
    if errors:
        with open(err_path, "w", encoding="utf-8") as fh:
            for n, tb in errors:
                fh.write(f"=== {n}\n{tb}\n")
    elif os.path.exists(err_path):
        os.remove(err_path)
    zpath = None
    if not a.no_zip:
        zpath = os.path.join(post, f"{did}_risultati_per_report.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in [rj, sm, err_path] + figures + tables + products:
                if f and os.path.exists(f):
                    zf.write(f, os.path.relpath(f, out))
            for f in sorted(glob.glob(os.path.join(pre, "*")) + glob.glob(os.path.join(pre, "figure_qc", "*"))):
                if os.path.isfile(f) and not f.endswith("_L2_wave_bursts.nc"):
                    zf.write(f, os.path.relpath(f, out))
    print(f"\nPost-processing completato: {post}")
    print(f"  {len(products)} prodotti L3, {len(tables)} tabelle, {len(figures)} figure, risultati.json")
    if errors:
        print(f"ATTENZIONE: {len(errors)} passi con errori, dettagli in {err_path}")
    if zpath:
        print(f"File da inviare per il report: {zpath} ({os.path.getsize(zpath) / 2**20:.1f} MB)")
    return dict(id=did, dir=post, figures=figures, tables=tables, products=products, errors=errors, zip=zpath,
                results=res)


def build_parser(prog: str = "postprocessing.py") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", required=True, help="cartella di output del deployment (quella del pre-processing)")
    p.add_argument("--soglia-mareggiata", type=float, default=1.0, help="soglia di Hm0 per le mareggiate [m]")
    g = p.add_argument_group("onde")
    g.add_argument("--nperseg", type=int, default=256, help="campioni per segmento dello spettro di Welch")
    g.add_argument("--fmin", type=float, default=0.04, help="frequenza minima [Hz]")
    g.add_argument("--fmax", type=float, default=0.6, help="frequenza massima per la direzione [Hz]")
    g.add_argument("--kp2-min", type=float, default=0.05, help="soglia della funzione di trasferimento della pressione")
    g.add_argument("--hm0-dir-min", type=float, default=0.25, help="Hm0 sotto cui la direzione e' sospetta [m]")
    p.add_argument("--no-zip", action="store_true", help="non creare lo zip dei risultati")
    return p


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
