"""Conversione di uno o più file PD0 in NetCDF (un insieme di file "part" per stream)
e lettura lazy degli stream convertiti."""
from __future__ import annotations

import glob
import json
import os
import time as _time

import numpy as np
import pandas as pd
import xarray as xr

from .pd0 import index_file, iter_decode, to_dataset, encoding_for, parse_byte_range

PD0_EXT = (".pd0", ".000", ".001", ".002", ".003", ".004", ".005")


def expand_inputs(inputs) -> list:
    """Accetta file, cartelle e pattern glob."""
    files = []
    for x in inputs:
        if os.path.isdir(x):
            for f in sorted(os.listdir(x)):
                if f.lower().endswith(PD0_EXT):
                    files.append(os.path.join(x, f))
        elif any(c in x for c in "*?["):
            files.extend(sorted(glob.glob(x)))
        else:
            files.append(x)
    return files


def _concat_dicts(parts: list) -> dict:
    out = {}
    for k in parts[0]:
        v = parts[0][k]
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            out[k] = np.concatenate([p[k] for p in parts])
        else:
            out[k] = v
    return out


def convert_files(files, outdir: str, chunk_bytes: float = 150e6, verbose: bool = True,
                  byte_range: str | None = None) -> dict:
    """Converte i file PD0 in ``outdir/raw/<stream>/*.nc`` e scrive ``outdir/inventory.json``.

    ``byte_range`` (es. "25%:75%" o "100:3600" in MB) limita ogni file a una porzione."""
    files = expand_inputs(files)
    if not files:
        raise FileNotFoundError("nessun file PD0 trovato")
    os.makedirs(os.path.join(outdir, "raw"), exist_ok=True)
    inv = dict(files={}, streams={})
    for f in files:
        t0 = _time.time()
        if verbose:
            print(f"[{os.path.basename(f)}] {os.path.getsize(f) / 2**20:,.1f} MB: indicizzazione...", flush=True)
        b0, b1 = parse_byte_range(byte_range, os.path.getsize(f))
        if verbose and byte_range:
            print(f"  porzione {byte_range}: da {b0 / 2**20:,.1f} a {b1 / 2**20:,.1f} MB", flush=True)
        idx = index_file(f, verbose=verbose, start_byte=b0, end_byte=b1)
        stem = os.path.splitext(os.path.basename(f))[0]
        counter = {}
        written = []
        for name, cfg, d in iter_decode(idx, chunk_bytes=chunk_bytes):
            ds = to_dataset(d, source=f)
            ds = ds.isel(time=np.flatnonzero(~np.isnat(ds.time.values)))
            if ds.sizes["time"] == 0:
                continue
            k = counter.get(name, 0)
            counter[name] = k + 1
            sdir = os.path.join(outdir, "raw", name)
            os.makedirs(sdir, exist_ok=True)
            out = os.path.join(sdir, f"{stem}_{k:04d}.nc")
            ds.to_netcdf(out, encoding=encoding_for(ds))
            written.append(out)
            s = inv["streams"].setdefault(name, dict(config=cfg, n=0, t_first=None, t_last=None,
                                                     variables=sorted(ds.data_vars), attrs={}))
            s["n"] += int(ds.sizes["time"])
            tt = ds.time.values
            s["t_first"] = str(min(np.datetime64(s["t_first"]) if s["t_first"] else tt.min(), tt.min()))
            s["t_last"] = str(max(np.datetime64(s["t_last"]) if s["t_last"] else tt.max(), tt.max()))
            s["attrs"] = {kk: (vv if isinstance(vv, (int, float, str)) else str(vv))
                          for kk, vv in ds.attrs.items() if kk != "source_file"}
        st = dict(idx.stats)
        st["n_resync"] = idx.n_resync
        st["layouts"] = [dict(nbytes=l.nbytes, count=l.count, data_types=l.names()) for l in idx.layouts]
        st["parts"] = len(written)
        st["seconds"] = round(_time.time() - t0, 1)
        inv["files"][f] = st
        if verbose:
            print(f"  {st['n_ensembles']:,} ensemble, checksum errate {st['bad_checksum']}, "
                  f"risincronizzazioni {idx.n_resync}, byte non attribuiti {st['bytes_skipped']:,} "
                  f"({st['seconds']} s)", flush=True)
    classify_streams(outdir, inv)
    with open(os.path.join(outdir, "inventory.json"), "w") as fh:
        json.dump(inv, fh, indent=2, default=str)
    if verbose:
        print_inventory(inv)
    return inv


def classify_streams(outdir: str, inv: dict) -> None:
    """Etichetta gli stream: 'waves' = campionamento rapido a burst, 'currents' = profili mediati."""
    for name, s in inv["streams"].items():
        t = StreamReader(outdir, name).times()
        dt = np.diff(t).astype("timedelta64[ms]").astype(float) / 1000.0
        med = float(np.median(dt)) if len(dt) else np.nan
        s["median_dt_s"] = med
        s["n_gaps"] = int((dt > max(5 * med, med + 1.0)).sum()) if len(dt) else 0
        s["label"] = "waves" if (np.isfinite(med) and med <= 2.5 and s["n_gaps"] > 0) else "currents"
    # se più stream "currents", quello con più ensemble è il principale
    cur = [n for n, s in inv["streams"].items() if s["label"] == "currents"]
    if len(cur) > 1:
        main = max(cur, key=lambda n: inv["streams"][n]["n"])
        for n in cur:
            if n != main:
                inv["streams"][n]["label"] = "other"
    wav = [n for n, s in inv["streams"].items() if s["label"] == "waves"]
    if len(wav) > 1:
        main = max(wav, key=lambda n: inv["streams"][n]["n"])
        for n in wav:
            if n != main:
                inv["streams"][n]["label"] = "other"


def print_inventory(inv: dict) -> None:
    print("\nStream trovati:")
    print(f"  {'nome':28s} {'tipo':9s} {'ensemble':>10s} {'dt med [s]':>10s}  periodo")
    for n, s in inv["streams"].items():
        print(f"  {n:28s} {s.get('label', '?'):9s} {s['n']:>10,} {s.get('median_dt_s', float('nan')):>10.2f}"
              f"  {s['t_first'][:19]} -> {s['t_last'][:19]}")
        a = s["attrs"]
        print(f"      coord={a.get('coord_system')} orient={a.get('orientation')} "
              f"beam_angle={a.get('beam_angle_deg')} celle={a.get('n_cells')} x {a.get('cell_size_cm')} cm, "
              f"bin1={a.get('bin1_dist_cm')} cm, variabili: {', '.join(s['variables'])}")


def load_inventory(outdir: str) -> dict:
    with open(os.path.join(outdir, "inventory.json")) as fh:
        return json.load(fh)


def pick_stream(outdir: str, label: str, name: str | None = None) -> str | None:
    if name:
        return name
    inv = load_inventory(outdir)
    for n, s in inv["streams"].items():
        if s.get("label") == label:
            return n
    return None


class StreamReader:
    """Accesso lazy a uno stream convertito (molti file part) per finestre temporali."""

    def __init__(self, outdir: str, stream: str):
        self.dir = os.path.join(outdir, "raw", stream)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.nc")))
        if not self.files:
            raise FileNotFoundError(f"nessun dato per lo stream {stream} in {self.dir}")
        self._ds = [xr.open_dataset(f, cache=False) for f in self.files]
        self._t = [d.time.values for d in self._ds]

    @property
    def attrs(self) -> dict:
        return dict(self._ds[0].attrs)

    def times(self) -> np.ndarray:
        return np.unique(np.concatenate(self._t))  # ordinati, senza duplicati

    def variables(self) -> list:
        return list(self._ds[0].data_vars)

    def load(self, variables=None, t0=None, t1=None, isel=None) -> xr.Dataset:
        """Carica le variabili richieste nell'intervallo [t0, t1), ordinate nel tempo."""
        out = []
        for d, t in zip(self._ds, self._t):
            m = np.ones(len(t), bool)
            if t0 is not None:
                m &= t >= np.datetime64(t0)
            if t1 is not None:
                m &= t < np.datetime64(t1)
            if not m.any():
                continue
            sub = d if variables is None else d[[v for v in variables if v in d]]
            if isel:
                sub = sub.isel({k: v for k, v in isel.items() if k in sub.dims})
            out.append(sub.isel(time=np.flatnonzero(m)).load())
        if not out:
            return xr.Dataset()
        ds = xr.concat(out, dim="time", data_vars="all", coords="minimal", compat="override",
                       combine_attrs="override") if len(out) > 1 else out[0]
        ds = ds.sortby("time")
        _, iu = np.unique(ds.time.values, return_index=True)
        if len(iu) < ds.sizes["time"]:
            ds = ds.isel(time=iu)
        return ds

    def close(self):
        for d in self._ds:
            d.close()


def burst_summary(t: np.ndarray) -> dict:
    """Statistiche dei burst (numero, durata, intervallo) da un vettore di tempi."""
    from .waves import find_bursts
    t = np.unique(t[~np.isnat(t)])
    b, med = find_bursts(t)
    n = b[:, 1] - b[:, 0]
    starts = t[b[:, 0]]
    rep = np.diff(starts).astype("timedelta64[s]").astype(float) / 60 if len(starts) > 1 else np.array([np.nan])
    return dict(n_bursts=len(b), dt_s=med, pings_median=int(np.median(n)), pings_min=int(n.min()),
                pings_max=int(n.max()), duration_median_s=float(np.median(n) * med),
                interval_median_min=float(np.median(rep)), interval_min_min=float(np.min(rep)),
                interval_max_min=float(np.max(rep)), lengths=n)


def inspect_files(files, max_mb: float | None = None, byte_range: str | None = None) -> None:
    """Ispezione veloce: layout, data type, stream, periodo e campionamento (solo leader)."""
    from .pd0 import describe_config, _decode_fixed, _v
    from .transform import theoretical_matrix
    for f in expand_inputs(files):
        print(f"\n=== {f} ({os.path.getsize(f) / 2**20:,.1f} MB)")
        b0, b1 = parse_byte_range(byte_range, os.path.getsize(f))
        idx = index_file(f, max_bytes=None if max_mb is None else b0 + max_mb * 2**20, verbose=False,
                         start_byte=b0, end_byte=b1)
        st = idx.stats
        if st.get("partial"):
            print(f"(analizzata solo la porzione {st['start_byte'] / 2**20:,.1f} - "
                  f"{st['end_byte'] / 2**20:,.1f} MB)")
        if idx.n_resync:
            print(f"  ATTENZIONE: {idx.n_resync} risincronizzazioni, "
                  f"{st['bytes_skipped']:,} byte non riconosciuti come ensemble validi")
        print(f"ensemble: {st['n_ensembles']:,}  risincronizzazioni: {idx.n_resync}  "
              f"byte non attribuiti: {st['bytes_skipped']:,}")
        for i, l in enumerate(idx.layouts):
            print(f"  layout {i}: {l.count:,} ens x {l.nbytes + 2} B -> "
                  + ", ".join(f"{n}({s}B)" for n, s in zip(l.names(), l.sizes)))
        # configurazione e matrice di trasformazione dal primo ensemble
        lay = max(idx.layouts, key=lambda l: l.count)
        p0 = int(idx.positions[idx.layout_id == idx.layouts.index(lay)][0])
        M = np.frombuffer(idx._mm, np.uint8, count=lay.nbytes + 2, offset=p0)[None, :]
        fl = {k: int(v[0]) for k, v in _decode_fixed(M, *lay.find("fixed_leader")).items()}
        cfg = describe_config(fl)
        print(f"  strumento: s/n {fl.get('serial')}, fw {fl['fw_ver']}.{fl['fw_rev']}, "
              f"orientamento {cfg['orientation']}, angolo beam {cfg['beam_angle_deg']} gradi, "
              f"coordinate registrate: {cfg['coord_system'].upper()}"
              + (" (tilt usati)" if cfg["tilts_used"] else "") + (", bin mapping" if cfg["bin_mapping"] else ""))
        tm = lay.find("transformation_matrix")
        if tm is not None:
            T = np.frombuffer(
                M[0, tm[0] + 2:tm[0] + 34].tobytes(), "<i2").reshape(4, 4) / 10000.0
            Tt = theoretical_matrix(cfg["beam_angle_deg"], bool(cfg["convex"]))
            print("  matrice beam->strumento registrata (righe X, Y, Z, err)  |  teorica RDI:")
            for r1, r2 in zip(T, Tt):
                print("     " + " ".join(f"{x:7.4f}" for x in r1) + "   |  " + " ".join(f"{x:7.4f}" for x in r2))
        streams = {}
        for name, scfg, d in iter_decode(idx, leaders_only=True):
            streams.setdefault(name, dict(cfg=scfg, t=[], p=[], h=[], pr=[]))
            streams[name]["t"].append(d["time"])
            streams[name]["h"].append(d["heading"])
            streams[name]["pr"].append(np.hypot(d["pitch"], d["roll"]))
            if "pressure" in d:
                streams[name]["p"].append(d["pressure"])
        for name, s in streams.items():
            t = np.sort(np.concatenate(s["t"]))
            t = t[~np.isnat(t)]
            p = np.concatenate(s["p"]) if s["p"] else np.array([np.nan])
            h = np.concatenate(s["h"])
            tilt = np.concatenate(s["pr"])
            print(f"  stream {name}: {len(t):,} ens, {t[0]} -> {t[-1]}")
            print(f"      config: {s['cfg']}")
            print(f"      pressione {np.nanmin(p):.2f}..{np.nanmax(p):.2f} dbar, heading "
                  f"{np.nanmin(h):.1f}..{np.nanmax(h):.1f} gradi, tilt max {np.nanmax(tilt):.1f} gradi")
            bs = burst_summary(t)
            print(f"      dt mediano {bs['dt_s']:.2f} s; burst: {bs['n_bursts']}, ping per burst "
                  f"mediana {bs['pings_median']} (min {bs['pings_min']}, max {bs['pings_max']}) = "
                  f"{bs['duration_median_s'] / 60:.1f} min; intervallo fra inizi burst mediana "
                  f"{bs['interval_median_min']:.1f} min (min {bs['interval_min_min']:.1f}, "
                  f"max {bs['interval_max_min']:.1f})")
            vals, cnt = np.unique(bs["lengths"], return_counts=True)
            top = np.argsort(cnt)[::-1][:5]
            print("      lunghezze burst piu' frequenti (ping x volte): "
                  + ", ".join(f"{vals[i]}x{cnt[i]}" for i in top))
