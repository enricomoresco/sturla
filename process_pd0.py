#!/usr/bin/env python
"""Preprocessing ADCP Sentinel V (.pd0): conversione, QC correnti, onde, grafici.

Esempi:
  python process_pd0.py inspect dati/*.pd0 --max-mb 200
  python process_pd0.py convert dati/ -o out
  python process_pd0.py currents -o out --declination 2.5
  python process_pd0.py waves -o out --hab 0.6
  python process_pd0.py all dati/ -o out --hab 0.6
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr

from adcp_pd0 import (convert_files, inspect_files, StreamReader, pick_stream, qc_currents,
                      analyze_waves)
from adcp_pd0.currents import average_pings
from adcp_pd0 import plots


def _aux_dataset(reader, max_points=200_000):
    t = reader.times()
    ds = reader.load(["pressure", "temperature", "heading", "pitch", "roll"])
    if ds.sizes["time"] > max_points:
        ds = ds.isel(time=slice(None, None, ds.sizes["time"] // max_points + 1))
    return ds


def _deployment(a, reader):
    from adcp_pd0.deployment import get_deployment
    return get_deployment(a.out, reader, start=a.start, end=a.end, redetect=a.redetect, p_min=a.p_min)


def run_currents(a):
    name = pick_stream(a.out, "currents", a.currents_stream)
    if name is not None:
        print(f"\nCorrenti: stream {name}")
        r = StreamReader(a.out, name)
        ds = r.load()
    else:
        # nessuno stream di profili mediati: si mediano i ping singoli dello stream onde
        name = pick_stream(a.out, "waves", a.waves_stream)
        if name is None:
            print("Nessuno stream utilizzabile per le correnti.")
            return None
        print(f"\nCorrenti: medie dei ping singoli dello stream {name} "
              + (f"su finestre di {a.avg_minutes} min" if a.avg_minutes else "su ogni burst"))
        r = StreamReader(a.out, name)
        dep = _deployment(a, r)
        ds = average_pings(r, window_min=a.avg_minutes, corr_min=a.corr_min,
                           orientation=a.orientation)
        ds.to_netcdf(os.path.join(a.out, "currents_avg.nc"))
        if "vb_echo" in r.variables():
            from adcp_pd0.diagnostics import check_transform
            print("  controllo trasformazione con la superficie del beam verticale (dη/dt):")
            check_transform(r, os.path.join(a.out, "figures"), orientation=a.orientation,
                            corr_min=a.corr_min, deployment=dep)
    dep = _deployment(a, r)
    qc = qc_currents(ds, corr_min=a.corr_min, pg_min=a.pg_min, err_max=a.err_max,
                     tilt_max=a.tilt_max, p_min=a.p_min, declination=a.declination,
                     orientation=a.orientation, deployment=dep, trim=a.trim)
    path = os.path.join(a.out, "currents_qc.nc")
    enc = {k: dict(zlib=True, complevel=1) for k in qc.data_vars}
    qc.to_netcdf(path, encoding=enc)
    ip = qc.in_position.values.astype(bool)
    good = np.isfinite(qc.u.values[ip]).mean() * 100 if ip.any() else 0.0
    print(f"  {qc.sizes['time']:,} profili ({ip.sum():,} in posizione, {(~ip).sum():,} fuori posizione "
          f"{'eliminati' if a.trim else 'con flag 4'}); celle valide in posizione {good:.1f}% -> {path}")
    r.close()
    return qc


def run_waves(a):
    name = pick_stream(a.out, "waves", a.waves_stream)
    if name is None:
        print("Nessuno stream di onde trovato (nessuno stream a burst con dt <= 2.5 s).")
        return None
    print(f"\nOnde: stream {name}")
    r = StreamReader(a.out, name)
    w = analyze_waves(r, hab=a.hab, p_atm=a.p_atm, nperseg=a.nperseg, fmin=a.fmin, fmax=a.fmax,
                      kp2_min=a.kp2_min, use_vb=not a.no_vb, p_min=a.p_min, puv_cell=a.puv_cell,
                      orientation=a.orientation, declination=a.declination, dir_ref=a.dir_ref)
    from adcp_pd0.deployment import in_position
    dep = _deployment(a, r)
    pos = in_position(w.time.values, dep)
    if "depth_sensor" in w:
        pos &= np.nan_to_num(w.depth_sensor.values) * 1025 * 9.81 / 1e4 >= a.p_min
    if a.trim:
        w = w.isel(time=np.flatnonzero(pos))
        pos = pos[pos]
    else:
        # fuori posizione: valori rimossi, la riga resta (flag 4 nell'archivio)
        for k in w.data_vars:
            if "time" in w[k].dims and k not in ("n", "duration_s", "depth", "depth_sensor", "p_std"):
                w[k] = w[k].where(xr.DataArray(pos, dims="time"))
    w["in_position"] = ("time", pos.astype(np.uint8), dict(long_name="1 = strumento in posizione sul fondo"))
    w.attrs.update(deployment_start=dep["start"], deployment_end=dep["end"],
                   deployment_method=dep.get("method", ""))
    path = os.path.join(a.out, "waves_bulk.nc")
    w.to_netcdf(path)
    print(f"  {w.sizes['time']} burst -> {path}")
    if "surface_range_vb" in w:
        dr = float(np.nanmedian(w.surface_range_vb - w.depth_sensor))
        ff = float(np.nanmedian(w.surface_found_frac)) if "surface_found_frac" in w else np.nan
        print(f"  controllo beam verticale: superficie trovata nel {ff * 100:.0f}% dei ping (mediana); "
              f"distanza superficie - profondita' da pressione = {dr:+.2f} m "
              "(atteso: piccolo, pari alla distanza verticale fra sensore di pressione e trasduttore)")
    r.close()
    return w


def run_plots(a, qc=None, w=None):
    import xarray as xr
    from adcp_pd0.convert import load_inventory
    if qc is None and os.path.exists(os.path.join(a.out, "currents_qc.nc")):
        qc = xr.load_dataset(os.path.join(a.out, "currents_qc.nc"))
    if w is None and os.path.exists(os.path.join(a.out, "waves_bulk.nc")):
        w = xr.load_dataset(os.path.join(a.out, "waves_bulk.nc"))
    aux = {}
    for name, s in load_inventory(a.out)["streams"].items():
        if s.get("label") in ("currents", "waves"):
            r = StreamReader(a.out, name)
            aux[name] = _aux_dataset(r)
            r.close()
    print("\nGrafici:")
    plots.plot_all(a.out, qc=qc, waves=w, aux=aux)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="ispezione veloce dei file (nessun output scritto)")
    pi.add_argument("inputs", nargs="+")
    pi.add_argument("--max-mb", type=float, default=None, help="analizza solo N MB (dall'inizio della porzione)")
    RANGE_HELP = ("porzione di ogni file da usare, INIZIO:FINE in %% o in MB, es. 25%%:75%% "
                  "(meta' centrale), 50:3650 (MB), 5%%: (salta il primo 5%%)")
    pi.add_argument("--range", dest="byte_range", default=None, help=RANGE_HELP)

    def common(sp):
        sp.add_argument("-o", "--out", required=True, help="cartella di output")
        sp.add_argument("--start", default=None,
                        help="correzione manuale dell'inizio del periodo in posizione (default: automatico)")
        sp.add_argument("--end", default=None, help="correzione manuale della fine del periodo in posizione")
        sp.add_argument("--trim", action="store_true",
                        help="elimina i dati fuori posizione invece di marcarli con flag 4")
        sp.add_argument("--redetect", action="store_true",
                        help="ricalcola il periodo in posizione (ignora out/deployment.json)")
        sp.add_argument("--orientation", choices=["up", "down"], default=None,
                        help="forza l'orientamento (default: dal fixed leader)")
        sp.add_argument("--declination", type=float, default=0.0,
                        help="declinazione magnetica [°, positiva verso Est] se non già impostata (EB)")
        g = sp.add_argument_group("correnti")
        g.add_argument("--currents-stream", default=None)
        g.add_argument("--corr-min", type=int, default=64)
        g.add_argument("--pg-min", type=float, default=50)
        g.add_argument("--err-max", type=float, default=0.2, help="m/s")
        g.add_argument("--tilt-max", type=float, default=20)
        g.add_argument("--p-min", type=float, default=1.0, help="dbar: sotto = fuori acqua")
        g.add_argument("--avg-minutes", type=float, default=None,
                       help="se le correnti si ricavano dai ping singoli: finestre di N minuti "
                            "(default: un profilo per burst)")
        g = sp.add_argument_group("onde")
        g.add_argument("--waves-stream", default=None)
        g.add_argument("--hab", type=float, default=0.5, help="altezza sensore sul fondo [m]")
        g.add_argument("--p-atm", type=float, default=0.0, help="offset pressione atmosferica [dbar]")
        g.add_argument("--nperseg", type=int, default=256)
        g.add_argument("--fmin", type=float, default=0.04)
        g.add_argument("--fmax", type=float, default=0.6)
        g.add_argument("--kp2-min", type=float, default=0.05)
        g.add_argument("--puv-cell", type=lambda x: x if x == "auto" else int(x), default=None,
                       help="cella (0-based) per la direzione, oppure 'auto'; default: nessuna")
        g.add_argument("--no-vb", action="store_true", help="non usare il beam verticale")
        g.add_argument("--dir-ref", choices=["auto", "vb", "p"], default="auto",
                       help="riferimento per la direzione: superficie dal beam verticale o pressione")

    pc = sub.add_parser("convert", help="PD0 -> NetCDF")
    pc.add_argument("inputs", nargs="+")
    pc.add_argument("-o", "--out", required=True)
    pc.add_argument("--range", dest="byte_range", default=None, help=RANGE_HELP)
    for n in ("currents", "waves", "plots"):
        common(sub.add_parser(n))
    common(sub.add_parser("deployment", help="individua il periodo in posizione (deployment.json + grafico)"))
    par = sub.add_parser("archive", help="file d'archivio NetCDF (CF) e CSV")
    common(par)
    par.add_argument("--metadata", default=None, help="file JSON di metadati (vedi metadata_template.json)")
    pk = sub.add_parser("check", help="solo controllo trasformazione (burst piu' energetici)")
    common(pk)
    pk.add_argument("--n-bursts", type=int, default=10)
    pa = sub.add_parser("all", help="convert + currents + waves + plots")
    pa.add_argument("inputs", nargs="+")
    pa.add_argument("--range", dest="byte_range", default=None, help=RANGE_HELP)
    pa.add_argument("--metadata", default=None, help="file JSON di metadati per l'archivio")
    common(pa)

    a = p.parse_args(argv)
    if a.cmd == "inspect":
        inspect_files(a.inputs, a.max_mb, a.byte_range)
    elif a.cmd == "convert":
        convert_files(a.inputs, a.out, byte_range=a.byte_range)
    elif a.cmd == "currents":
        run_currents(a)
    elif a.cmd == "waves":
        run_waves(a)
    elif a.cmd == "plots":
        run_plots(a)
    elif a.cmd == "archive":
        from adcp_pd0.archive import load_metadata, write_archive
        write_archive(a.out, load_metadata(a.metadata), hab=a.hab)
    elif a.cmd == "deployment":
        from adcp_pd0.deployment import get_deployment
        name = pick_stream(a.out, "currents", a.currents_stream) or pick_stream(a.out, "waves", a.waves_stream)
        get_deployment(a.out, StreamReader(a.out, name), redetect=True, p_min=a.p_min)
    elif a.cmd == "check":
        from adcp_pd0.diagnostics import check_transform
        r = StreamReader(a.out, pick_stream(a.out, "waves", a.waves_stream))
        check_transform(r, os.path.join(a.out, "figures"), n_bursts=a.n_bursts,
                        orientation=a.orientation, corr_min=a.corr_min, deployment=_deployment(a, r))
    elif a.cmd == "all":
        convert_files(a.inputs, a.out, byte_range=a.byte_range)
        qc = run_currents(a)
        w = run_waves(a)
        run_plots(a, qc, w)
        from adcp_pd0.archive import load_metadata, write_archive
        write_archive(a.out, load_metadata(a.metadata), hab=a.hab)


if __name__ == "__main__":
    sys.exit(main())
