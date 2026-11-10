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

from .pd0 import index_file, iter_decode, to_dataset, encoding_for

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


def convert_files(files, outdir: str, chunk_bytes: float = 150e6, verbose: bool = True) -> dict:
    """Converte i file PD0 in ``outdir/raw/<stream>/*.nc`` e scrive ``outdir/inventory.json``."""
    files = expand_inputs(files)
    if not files:
        raise FileNotFoundError("nessun file PD0 trovato")
    os.makedirs(os.path.join(outdir, "raw"), exist_ok=True)
    inv = dict(files={}, streams={})
    for f in files:
        t0 = _time.time()
        if verbose:
            print(f"[{os.path.basename(f)}] {os.path.getsize(f) / 2**20:,.1f} MB: indicizzazione...", flush=True)
        idx = index_file(f, verbose=verbose)
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


def inspect_files(files, max_mb: float | None = None) -> None:
    """Ispezione veloce: layout, data type, stream, periodo e campionamento (solo leader)."""
    for f in expand_inputs(files):
        print(f"\n=== {f} ({os.path.getsize(f) / 2**20:,.1f} MB)")
        idx = index_file(f, max_bytes=None if max_mb is None else max_mb * 2**20, verbose=False)
        st = idx.stats
        print(f"ensemble: {st['n_ensembles']:,}  risincronizzazioni: {idx.n_resync}  "
              f"byte non attribuiti: {st['bytes_skipped']:,}")
        for i, l in enumerate(idx.layouts):
            print(f"  layout {i}: {l.count:,} ens x {l.nbytes + 2} B -> "
                  + ", ".join(f"{n}({s}B)" for n, s in zip(l.names(), l.sizes)))
        streams = {}
        for name, cfg, d in iter_decode(idx, leaders_only=True):
            streams.setdefault(name, dict(cfg=cfg, t=[], p=[]))
            streams[name]["t"].append(d["time"])
            if "pressure" in d:
                streams[name]["p"].append(d["pressure"])
        for name, s in streams.items():
            t = np.sort(np.concatenate(s["t"]))
            t = t[~np.isnat(t)]
            dt = np.diff(t).astype("timedelta64[ms]").astype(float) / 1000
            med = np.median(dt) if len(dt) else np.nan
            gaps = int((dt > max(5 * med, med + 1)).sum()) if len(dt) else 0
            p = np.concatenate(s["p"]) if s["p"] else np.array([np.nan])
            print(f"  stream {name}: {len(t):,} ens, {t[0]} -> {t[-1]}, dt mediano {med:.2f} s, "
                  f"interruzioni (burst) {gaps}, pressione {np.nanmin(p):.2f}..{np.nanmax(p):.2f} dbar")
            print(f"      config: {s['cfg']}")
