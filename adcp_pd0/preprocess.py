"""Pre-processing: dal file PD0 ai dati puliti in NetCDF (livello L2).

  python preprocessing.py file.pd0 -o out [--metadata metadata.json]

Passi: lettura del PD0 con controllo delle checksum (out/raw, livello L1); periodo in cui lo
strumento è in posizione e movimenti del telaio; correnti (medie orarie, controlli e flag);
temperatura oraria; serie pulite dei burst d'onda; controllo della trasformazione; rapporto.
Nessuna analisi: spettri, statistiche e grafici sono nel post-processing.

Risultati in out/preprocessing/:
  <ID>_L2_currents.nc, <ID>_L2_temperature.nc, <ID>_L2_wave_bursts.nc,
  <ID>_RAPPORTO_PREPROCESSING.md, deployment.json, inventory.json, metadata.json, figure_qc/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import traceback

import numpy as np

from .archive import (TIME_ENC, _encoding, currents_archive, deployment_id, hourly_temperature,
                      load_metadata, temperature_archive, wave_bursts_archive)
from .convert import StreamReader, convert_files, expand_inputs, load_inventory, pick_stream, print_inventory
from .currents import average_pings, qc_currents
from .deployment import get_deployment
from .waves import extract_wave_bursts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_LEFTOVERS = ("archive", "figures", "currents_avg.nc", "currents_qc.nc", "waves_bulk.nc",
                 "check_transform.json")


def resolve_metadata(path: str | None, out: str):
    """Metadati: file indicato, altrimenti metadata.json (cartella corrente o di output),
    altrimenti il modello non compilato."""
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"file di metadati non trovato: {path}\n  crearlo copiando metadata_template.json "
                             "e compilandolo, oppure togliere --metadata")
        return load_metadata(path), path
    for cand in ("metadata.json", os.path.join(out, "metadata.json")):
        if os.path.exists(cand):
            return load_metadata(cand), cand
    tpl = os.path.join(REPO, "metadata_template.json")
    if os.path.exists(tpl):
        return load_metadata(tpl), "metadata_template.json (modello NON compilato)"
    return {}, "nessuno"


def station_from_files(inv: dict) -> str | None:
    """Nome della stazione dal nome del file PD0 (il Sentinel V lo compone come <nome>_<data>.pd0)."""
    for f in inv.get("files", {}):
        m = re.search(r"[A-Za-z]{2,}", os.path.splitext(os.path.basename(f))[0])
        if m:
            return m.group(0).upper()
    return None


def ensure_converted(inputs, out: str, byte_range: str | None = None, force: bool = False,
                     verbose: bool = True) -> dict:
    """Converte i PD0 in out/raw, oppure riusa la conversione già presente se i file sono gli stessi."""
    inv_path = os.path.join(out, "inventory.json")
    files = expand_inputs(inputs) if inputs else []
    for f in files:
        if not os.path.exists(f):
            raise SystemExit(f"file PD0 non trovato: {f}")
    have = os.path.exists(inv_path) and os.path.isdir(os.path.join(out, "raw"))
    if files and have and not force and not byte_range:
        inv = load_inventory(out)
        old = {os.path.basename(k): v for k, v in inv.get("files", {}).items()}
        same = set(old) == {os.path.basename(f) for f in files} and all(
            not old[os.path.basename(f)].get("partial")
            and old[os.path.basename(f)].get("bytes_scanned") == os.path.getsize(f) for f in files)
        if same:
            if verbose:
                print(f"Conversione gia' presente in {os.path.join(out, 'raw')}: la riuso "
                      "(--riconverti per rifarla)")
                print_inventory(inv)
            return inv
    if files:
        return convert_files(files, out, byte_range=byte_range, verbose=verbose)
    if not have:
        raise SystemExit(f"nessun dato convertito in {out}: indicare il file PD0")
    if verbose:
        print(f"Uso la conversione presente in {os.path.join(out, 'raw')}")
    return load_inventory(out)


def _aux_dataset(reader, max_points=200_000):
    ds = reader.load(["pressure", "temperature", "heading", "pitch", "roll"])
    if ds.sizes["time"] > max_points:
        ds = ds.isel(time=slice(None, None, ds.sizes["time"] // max_points + 1))
    return ds


def _currents(out, reader, cur_name, dep, a):
    if cur_name is not None:
        print(f"  stream {cur_name}: profili gia' mediati dallo strumento")
        r = StreamReader(out, cur_name)
        ds = r.load()
        r.close()
    else:
        print("  medie dei ping singoli " + (f"su intervalli regolari di {a.avg_minutes:g} min"
                                             if a.avg_minutes else "su ogni burst"))
        ds = average_pings(reader, bin_minutes=a.avg_minutes or None, corr_min=a.corr_min,
                           orientation=a.orientation, deployment=dep, p_min=a.p_min, check_vb=False)
    qc = qc_currents(ds, corr_min=a.corr_min, pg_min=a.pg_min, err_max=a.err_max, tilt_max=a.tilt_max,
                     p_min=a.p_min, declination=a.declination, orientation=a.orientation, deployment=dep)
    ip = qc.in_position.values.astype(bool)
    good = np.isfinite(qc.u.values[ip]).mean() * 100 if ip.any() else 0.0
    print(f"  {qc.sizes['time']:,} profili ({ip.sum():,} in posizione, {(~ip).sum():,} fuori posizione con "
          f"flag 4); celle valide in posizione {good:.1f}%")
    return qc


def run(a) -> dict:
    out = a.out
    pre = os.path.join(out, "preprocessing")
    figq = os.path.join(pre, "figure_qc")
    md, md_src = resolve_metadata(a.metadata, out)
    hab = a.hab if a.hab is not None else float(md.get("instrument_height_above_bed_m") or 0.6)
    md = dict(md, instrument_height_above_bed_m=hab)
    print(f"Metadati: {md_src}; altezza del trasduttore sul fondale: {hab} m")
    inv = ensure_converted(a.inputs, out, a.byte_range, a.riconverti)
    if not md.get("station_id"):
        md["station_id"] = station_from_files(inv) or "ADCP"
        print(f"Stazione non indicata nei metadati: uso '{md['station_id']}' (dal nome del file PD0); "
              "per cambiarla compilare station_id in metadata.json")
    os.makedirs(figq, exist_ok=True)
    wav = pick_stream(out, "waves", a.waves_stream)
    cur = pick_stream(out, "currents", a.currents_stream)
    if wav is None and cur is None:
        raise SystemExit("nessuno stream utilizzabile nei dati convertiti")
    reader = StreamReader(out, wav or cur)

    print("\nPeriodo in posizione:")
    dep = get_deployment(out, reader, start=a.start, end=a.end, redetect=True,
                         fig_path=os.path.join(figq, "deployment.png"), p_min=a.p_min)
    did = deployment_id(md, dep)
    print(f"  identificativo del deployment: {did}")
    written, errors = [], []
    c = b = h = chk = None

    def save(ds, suffix, extra_enc=None):
        ds.attrs["deployment_id"] = did
        path = os.path.join(pre, f"{did}_{suffix}.nc")
        enc = _encoding(ds)
        enc.update(extra_enc or {})
        ds.to_netcdf(path, encoding=enc)
        written.append(path)
        print(f"  -> {path}")

    def fail(name):
        errors.append((name, traceback.format_exc()))
        print(f"  ERRORE in '{name}': {errors[-1][1].strip().splitlines()[-1]} (gli altri prodotti proseguono)")

    print("\nCorrenti:")
    try:
        qc = _currents(out, reader, cur, dep, a)
        c = currents_archive(qc, md, hab)
        save(c, "L2_currents")
    except Exception:
        fail("correnti")

    print("\nTemperatura oraria:")
    try:
        h = hourly_temperature(reader, dep, p_min=a.p_min)
        save(temperature_archive(h, md, reader.attrs, dep), "L2_temperature", {"time_bnds": dict(TIME_ENC)})
    except Exception:
        fail("temperatura")

    if wav is not None:
        print("\nSerie dei burst d'onda:")
        try:
            p_ref = dep.get("p_ref_dbar")
            bb = extract_wave_bursts(reader, hab=hab, p_atm=a.p_atm, min_samples=a.min_burst,
                                     max_samples=a.max_burst,
                                     use_vb=not a.no_vb, p_min=a.p_min, orientation=a.orientation,
                                     declination=a.declination, corr_min=a.corr_min,
                                     p_ref=(float(p_ref) - a.p_atm) if p_ref is not None else None)
            b = wave_bursts_archive(bb, md, dep, p_min=a.p_min)
            ok = np.isin(b["burst_qc"].values, (1, 3))
            print(f"  {b.sizes['time']:,} burst, {int(ok.sum()):,} utilizzabili (flag 1 o 3)")
            save(b, "L2_wave_bursts")
        except Exception:
            fail("burst d'onda")
        if {"vb_echo", "vel"} <= set(reader.variables()):
            print("\nControllo della trasformazione (superficie del beam verticale, dη/dt):")
            try:
                from .diagnostics import check_transform
                cj = os.path.join(pre, "check_transform.json")
                rows = check_transform(reader, figq, orientation=a.orientation, corr_min=a.corr_min,
                                       deployment=dep, summary_path=cj)
                if rows and os.path.exists(cj):
                    with open(cj, encoding="utf-8") as fh:
                        chk = json.load(fh)
                    written.append(cj)
            except Exception:
                fail("controllo trasformazione")
    else:
        print("\nNessuno stream a burst (ping singoli con interruzioni): niente serie d'onda.")

    try:
        from . import plots
        plots.plot_overview(_aux_dataset(reader), os.path.join(figq, "panoramica_sensori.png"),
                            f"Sensori ausiliari - {did}")
    except Exception:
        fail("panoramica sensori")

    # file di accompagnamento: periodo in posizione effettivo, inventario, metadati, guida
    for name, obj in (("deployment.json", dep), ("metadata.json", md)):
        with open(os.path.join(pre, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
        written.append(os.path.join(pre, name))
    if os.path.exists(os.path.join(out, "inventory.json")):
        shutil.copyfile(os.path.join(out, "inventory.json"), os.path.join(pre, "inventory.json"))
        written.append(os.path.join(pre, "inventory.json"))
    guide = os.path.join(REPO, "LEGGIMI_DATI.md")
    if os.path.exists(guide):
        shutil.copyfile(guide, os.path.join(pre, "LEGGIMI_DATI.md"))
        written.append(os.path.join(pre, "LEGGIMI_DATI.md"))
    try:
        from .report import write_report
        rep = write_report(os.path.join(pre, f"{did}_RAPPORTO_PREPROCESSING.md"), did, md, hab, inv, dep, chk,
                           c=c, b=b, h=h, files=list(written), reader=reader, p_atm=a.p_atm)
        written.append(rep)
    except Exception:
        fail("rapporto")
    reader.close()

    err_path = os.path.join(pre, "errori_preprocessing.txt")
    if errors:
        with open(err_path, "w", encoding="utf-8") as fh:
            for name, tb in errors:
                fh.write(f"=== {name}\n{tb}\n")
    elif os.path.exists(err_path):
        os.remove(err_path)
    print(f"\nPre-processing completato: {pre}")
    for f in written:
        print(f"  {os.path.basename(f)}")
    if errors:
        print(f"ATTENZIONE: {len(errors)} passi con errori, dettagli in {err_path}")
    old = [n for n in OLD_LEFTOVERS if os.path.exists(os.path.join(out, n))]
    if old:
        print(f"Nota: in {out} ci sono file di versioni precedenti non piu' usati, che si possono cancellare: "
              + ", ".join(old))
    print(f"Controllare: metadati letti da {md_src}; altezza del trasduttore sul fondale {hab} m"
          + (" (opzione --hab)" if a.hab is not None else "")
          + f"; stazione {md.get('station_id')}. Se un valore e' sbagliato, correggere metadata.json e rilanciare.")
    print(f"Prossimo passo: python postprocessing.py -o {out}")
    return dict(id=did, dir=pre, files=written, errors=errors)


def build_parser(prog: str = "preprocessing.py") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*",
                   help="file o cartelle PD0 (si possono omettere se la conversione e' gia' in --out)")
    p.add_argument("-o", "--out", required=True, help="cartella di output del deployment")
    p.add_argument("--metadata", default=None,
                   help="file JSON di metadati (default: metadata.json se esiste, altrimenti il modello)")
    p.add_argument("--hab", type=float, default=None,
                   help="altezza del trasduttore sul fondale [m] (default: dai metadati, altrimenti 0.6)")
    p.add_argument("--range", dest="byte_range", default=None,
                   help="porzione di ogni file, INIZIO:FINE in %% o MB (es. 45%%:55%%) per prove veloci")
    p.add_argument("--riconverti", action="store_true", help="rilegge il PD0 anche se out/raw esiste gia'")
    g = p.add_argument_group("periodo in posizione (default: automatico)")
    g.add_argument("--start", default=None, help="correzione manuale dell'inizio (es. 2026-07-10T00:00)")
    g.add_argument("--end", default=None, help="correzione manuale della fine")
    g = p.add_argument_group("correnti")
    g.add_argument("--currents-stream", default=None)
    g.add_argument("--avg-minutes", type=float, default=60,
                   help="medie dei ping singoli su intervalli regolari di N minuti (default 60; 0 = per burst)")
    g.add_argument("--corr-min", type=int, default=64, help="correlazione minima [counts]")
    g.add_argument("--pg-min", type=float, default=50, help="percent good minimo (solo dati in coordinate Earth)")
    g.add_argument("--err-max", type=float, default=0.2, help="velocita' d'errore massima [m/s]")
    g.add_argument("--tilt-max", type=float, default=20, help="inclinazione massima [gradi]")
    g.add_argument("--p-min", type=float, default=1.0, help="pressione sotto cui lo strumento e' fuori acqua [dbar]")
    g.add_argument("--declination", type=float, default=0.0,
                   help="declinazione magnetica [gradi, positiva verso Est] se non gia' nello strumento (EB)")
    g.add_argument("--orientation", choices=["up", "down"], default=None, help="forza l'orientamento")
    g = p.add_argument_group("burst d'onda")
    g.add_argument("--waves-stream", default=None)
    g.add_argument("--min-burst", type=int, default=256, help="numero minimo di ping di un burst d'onda")
    g.add_argument("--max-burst", type=int, default=4096,
                   help="i burst piu' lunghi (campionamento continuo) sono divisi in tratti di N ping")
    g.add_argument("--p-atm", type=float, default=0.0, help="offset di pressione da sottrarre [dbar]")
    g.add_argument("--no-vb", action="store_true", help="non usare il beam verticale")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
