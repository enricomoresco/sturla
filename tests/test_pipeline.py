import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic import make_file  # noqa: E402
from adcp_pd0 import (convert_files, StreamReader, pick_stream, qc_currents, extract_wave_bursts,  # noqa: E402
                      wave_spectra)
from adcp_pd0 import preprocess, postprocess  # noqa: E402
from adcp_pd0.pd0 import index_file  # noqa: E402
from adcp_pd0.transform import beam_to_inst, inst_to_earth  # noqa: E402
import process_pd0  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_pre(*args):
    """Pre-processing; fallisce se un passo ha dato errore."""
    r = preprocess.run(preprocess.build_parser().parse_args([str(a) for a in args]))
    assert not r["errors"], r["errors"]
    return r


def run_post(*args):
    """Post-processing; fallisce se un'analisi o una figura ha dato errore."""
    r = postprocess.run(postprocess.build_parser().parse_args([str(a) for a in args]))
    assert not r["errors"], r["errors"]
    return r


def one(folder, pattern):
    ff = sorted(folder.glob(pattern))
    assert len(ff) == 1, (pattern, ff)
    return ff[0]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("pd0")
    f = d / "test.pd0"
    truth = make_file(str(f), hours=6)
    out = d / "out"
    inv = convert_files([str(f)], str(out), verbose=False)
    return dict(file=str(f), out=str(out), truth=truth, inv=inv)


def test_index(data):
    idx = index_file(data["file"], verbose=False)
    # 1 ensemble con checksum errata viene comunque indicizzato (scartato in decodifica)
    assert idx.stats["n_ensembles"] == data["truth"]["n_events"]
    assert idx.n_resync >= 1


def test_streams(data):
    inv = data["inv"]
    labels = sorted(s["label"] for s in inv["streams"].values())
    assert labels == ["currents", "waves"]
    st = inv["files"][data["file"]]
    assert st["bad_checksum"] == 1
    total = sum(s["n"] for s in inv["streams"].values())
    assert total == data["truth"]["n_events"] - 1


def test_currents(data):
    name = pick_stream(data["out"], "currents")
    ds = StreamReader(data["out"], name).load()
    assert ds.vel.shape[1:] == (20, 4)
    assert ds.attrs["coord_system"] == "earth" and ds.attrs["orientation"] == "up"
    np.testing.assert_allclose(ds.heading.values, 45.0)
    qc = qc_currents(ds)
    assert np.isfinite(qc.u.values[:, :5]).all()
    assert np.isnan(qc.u.values[:, -1]).all()          # celle sopra la superficie
    assert abs(float(np.nanmean(qc.v.values[:, 5])) - 0.1 * ((6.5 / 15) ** (1 / 7))) < 0.005


def test_waves(data):
    """Burst puliti (PRE) -> spettri e parametri (POST), con uno stream gia' in coordinate Earth."""
    tr = data["truth"]
    r = StreamReader(data["out"], pick_stream(data["out"], "waves"))
    b = extract_wave_bursts(r, hab=tr["hab"], verbose=False)
    assert b.sizes["time"] == tr["hours"] and b.sizes["sample"] == 1024
    assert {"eta", "surface_range", "pressure", "u", "v"} <= set(b.data_vars)
    assert np.allclose(np.nanmean(b.eta.values, axis=1), 0, atol=1e-3)       # media del burst tolta
    w = wave_spectra(b)
    assert w.sizes["time"] == tr["hours"]
    assert np.allclose(w.Hm0, tr["Hs"], rtol=0.15)                  # dalla superficie (beam verticale)
    assert np.allclose(w.Hm0_pressure, tr["Hs"], rtol=0.08)         # dalla pressione, solo confronto
    assert np.allclose(w.Tp, tr["Tp"], rtol=0.06)
    assert np.allclose(w.Dp, tr["dir_from"], atol=5)
    assert np.allclose(w.water_depth, tr["depth"], atol=0.05)
    assert (w.wave_qc.values == 1).all() and (w.direction_qc.values == 1).all()
    assert (w.Ub.values > 0).all()
    # senza beam verticale i parametri vengono dalla pressione
    b2 = extract_wave_bursts(r, hab=tr["hab"], use_vb=False, verbose=False)
    assert "eta" not in b2
    w2 = wave_spectra(b2)
    assert np.allclose(w2.Hm0, tr["Hs"], rtol=0.08) and np.allclose(w2.Dp, tr["dir_from"], atol=5)
    assert "pressione" in w2.attrs["wave_method"]


def test_transform_roundtrip():
    # beam -> instrument: velocità orizzontale pura lungo X
    th = np.deg2rad(25)
    vx = 0.3
    b = np.array([vx * np.sin(th), -vx * np.sin(th), 0, 0])[None, None, :]
    vi = beam_to_inst(b, 25, True)
    np.testing.assert_allclose(vi[0, 0, :3], [vx, 0, 0], atol=1e-6)
    # rotazione: heading 90 con strumento verso il basso, X -> Est ruotato
    ve = inst_to_earth(np.array([[[0, 1.0, 0, 0]]]), [90.0], [0.0], [0.0], "down")
    np.testing.assert_allclose(ve[0, 0, :3], [1, 0, 0], atol=1e-6)


def test_pre_post_two_streams(data, tmp_path):
    """I due comandi su un file con uno stream di correnti gia' mediate e uno di burst d'onda."""
    out = tmp_path / "cli"
    assert process_pd0.main(["preprocess", data["file"], "-o", str(out), "--hab", "0.5"]) == 0
    pre = out / "preprocessing"
    for pat in ("*_L2_currents.nc", "*_L2_temperature.nc", "*_L2_wave_bursts.nc", "*_RAPPORTO_PREPROCESSING.md"):
        one(pre, pat)
    assert not (pre / "errori_preprocessing.txt").exists()
    assert process_pd0.main(["postprocess", "-o", str(out)]) == 0
    post = out / "postprocessing"
    assert not (post / "errori_postprocessing.txt").exists()
    w = xr.open_dataset(one(post, "*_L3_waves.nc"))
    assert np.allclose(w.Hm0, data["truth"]["Hs"], rtol=0.15)
    res = json.load(open(post / "risultati.json", encoding="utf-8"))
    assert res["errori"] == [] and res["onde"]["n_validi"] == data["truth"]["hours"]
    figs = os.listdir(post / "figure")
    for f in ("01_quadro_generale.png", "10_correnti_sezioni.png", "20_onde_serie.png", "21_onde_spettrogramma.png"):
        assert f in figs


@pytest.fixture(scope="module")
def sentinel(tmp_path_factory):
    from synthetic import make_sentinel_like
    d = tmp_path_factory.mktemp("sv")
    f = d / "sv.pd0"
    truth = make_sentinel_like(str(f))
    out = d / "out"
    inv = convert_files([str(f)], str(out), verbose=False)
    return dict(file=str(f), out=str(out), truth=truth, inv=inv)


def test_sentinel_like_currents(sentinel):
    from adcp_pd0.currents import average_pings
    inv, tr = sentinel["inv"], sentinel["truth"]
    assert [s["label"] for s in inv["streams"].values()] == ["waves"]
    name = pick_stream(sentinel["out"], "waves")
    r = StreamReader(sentinel["out"], name)
    assert r.attrs["coord_system"] == "beam"
    assert np.asarray(r.attrs["transformation_matrix"]).size == 16
    avg = average_pings(r, verbose=False)
    assert avg.sizes["time"] == tr["bursts"]
    assert avg.attrs["corr_w4beam_vs_vbeam"] > 0.9
    qc = qc_currents(avg)
    ok = ~tr["side"]
    np.testing.assert_allclose(np.nanmean(qc.u.values, 0)[ok][:-2], (tr["U"] * tr["prof"])[ok][:-2], atol=0.01)
    np.testing.assert_allclose(np.nanmean(qc.v.values, 0)[ok][:-2], (tr["V"] * tr["prof"])[ok][:-2], atol=0.01)


def test_sentinel_like_waves(sentinel):
    """Come il file reale: ping in coordinate beam; direzione stimata a tre quote."""
    tr = sentinel["truth"]
    r = StreamReader(sentinel["out"], pick_stream(sentinel["out"], "waves"))
    b = extract_wave_bursts(r, hab=tr["hab"], verbose=False)
    assert b.sizes["level"] == 3 and b.attrs["has_vertical_beam"] == 1
    frac = b.level_range.values / float(np.median(b.depth_sensor.values))
    assert np.allclose(frac, [0.5, 0.7, 0.85], atol=0.03)
    w = wave_spectra(b)
    assert np.allclose(w.Hm0, tr["Hs"], rtol=0.12)
    assert np.allclose(w.Tp, tr["Tp"], rtol=0.06)
    assert np.allclose(w.Dp, tr["dir_from"], atol=5) and np.allclose(w.Dm, tr["dir_from"], atol=5)
    assert np.allclose(w.Dp_level, tr["dir_from"], atol=6)          # stessa direzione a tutte le quote
    assert (w.dir_level_spread.values < 6).all() and (w.dir_coherence.values > 0.8).all()
    assert (w.direction_qc.values == 1).all()
    i = int(np.argmin(np.abs(w.frequency.values - 1 / tr["Tp"])))
    assert np.allclose(w.direction_f.values[:, i], tr["dir_from"], atol=6)
    # velocita' orbitale al fondo: teoria lineare per un'onda monocromatica, 2 sigma = sqrt(2) * ampiezza
    from adcp_pd0.waves import wavenumber
    k = float(wavenumber(np.array([1 / tr["Tp"]]), tr["depth"])[0])
    ub = np.sqrt(2) * (tr["Hs"] / (2 * np.sqrt(2))) * (2 * np.pi / tr["Tp"]) / np.sinh(k * tr["depth"])
    assert np.allclose(w.Ub, ub, rtol=0.15)


def test_trash_and_range(tmp_path):
    from synthetic import make_trashy
    from adcp_pd0.pd0 import parse_byte_range
    f = tmp_path / "trash.pd0"
    tr = make_trashy(str(f), bursts=6, depth=20.0)
    size = os.path.getsize(f)
    assert parse_byte_range("25%:75%", size) == (size // 4, int(size * 0.75))
    assert parse_byte_range(":1", size) == (0, 2**20)
    # 1) file intero: periodo in posizione trovato in automatico, il resto con flag 4
    out = tmp_path / "all"
    run_pre(f, "-o", out)
    pre = out / "preprocessing"
    dep = json.load(open(pre / "deployment.json"))
    t0, t1 = np.datetime64(dep["start"]), np.datetime64(dep["end"])
    assert np.datetime64(tr["t0"]) <= t0 <= np.datetime64(tr["t0"]) + np.timedelta64(30, "m")
    last = np.datetime64(tr["t_end"]) - np.timedelta64(30, "m") + np.timedelta64(512, "s")   # fine ultimo burst buono
    assert last - np.timedelta64(15, "m") <= t1 <= last
    assert (pre / "figure_qc" / "deployment.png").exists()
    # stazione non indicata nei metadati: presa dal nome del file (trash.pd0 -> TRASH)
    did = f"TRASH_{pd.Timestamp(dep['start']):%Y%m%d}_{pd.Timestamp(dep['end']):%Y%m%d}"
    c = xr.open_dataset(pre / f"{did}_L2_currents.nc")
    assert c.attrs["deployment_id"] == did and c.attrs["data_level"] == "L2"
    tt = c.time.values
    assert (np.diff(tt) == np.timedelta64(1, "h")).all()      # serie oraria regolare
    dq = c.deployment_qc.values
    ins = dq == 1
    assert ins.sum() == 3                                  # 00, 01, 02 (anche ore parziali)
    assert (dq == 4).sum() >= 3                            # aria + discesa/risalita marcati
    assert set(np.unique(dq)) <= {1, 4, 9}
    assert (c.vel_qc.values[~ins] >= 4).all()
    assert np.isnan(c.u.values[~ins]).all()
    assert np.isfinite(c.u.values[ins]).any(axis=1).all()
    # burst d'onda: quelli in posizione utilizzabili, gli altri con flag 4 ma presenti nel file
    b = xr.open_dataset(pre / f"{did}_L2_wave_bursts.nc")
    bq = b.burst_qc.values
    assert (bq == 1).sum() >= 4 and (bq[bq != 1] == 4).all() and (bq == 4).sum() >= 6
    assert ((b.burst_qc_cause.values & 1) > 0).sum() == (bq == 4).sum()
    # temperatura oraria: in posizione 14.5-15.5 °C (flag 1), in aria 25 °C (flag 4)
    th = xr.open_dataset(pre / f"{did}_L2_temperature.nc")
    q = th.temperature_qc.values
    assert (q == 1).sum() >= 2 and (q == 4).sum() >= 2
    assert np.all(np.abs(th.temperature.values[q == 1] - 15.0) <= 0.51)
    assert np.all(th.temperature.values[q == 4] > 14.4)
    assert th.temperature.attrs["cell_methods"].startswith("time: mean")
    assert "time_bnds" in th
    # 2) post-processing: onde solo dai burst in posizione
    run_post("-o", out)
    w = xr.open_dataset(out / "postprocessing" / f"{did}_L3_waves.nc")
    good = w.wave_qc.values == 1
    assert good.sum() == (bq == 1).sum() and (w.wave_qc.values[~good] == 4).all()
    assert np.allclose(w.Hm0.values[good], tr["Hs"], rtol=0.12)
    assert np.isnan(w.Hm0.values[~good]).all()
    # 3) solo la metà centrale del file
    out2 = tmp_path / "mid"
    run_pre(f, "-o", out2, "--range", "25%:75%")
    b2 = xr.load_dataset(one(out2 / "preprocessing", "*_L2_wave_bursts.nc"))
    w2 = wave_spectra(b2)
    ok2 = np.isfinite(w2.Hm0.values)
    assert ok2.sum() >= 1 and np.allclose(w2.Hm0.values[ok2], tr["Hs"], rtol=0.12)


def test_mixed_bursts(tmp_path):
    """Come il file reale: burst da 1024 ping (onde) alternati a burst da 180 ping."""
    from synthetic import make_sentinel_like
    from adcp_pd0.currents import average_pings
    f = tmp_path / "mix.pd0"
    tr = make_sentinel_like(str(f), bursts=9, interval_min=40, nping=[1024, 180, 180], depth=34.5)
    out = tmp_path / "out"
    convert_files([str(f)], str(out), verbose=False)
    r = StreamReader(str(out), pick_stream(str(out), "waves"))
    b = extract_wave_bursts(r, hab=0.5, verbose=False)
    assert b.sizes["time"] == 3 and b.attrs["n_short_bursts_skipped"] == 6
    w = wave_spectra(b)
    assert np.allclose(w.Hm0, 1.0, rtol=0.12) and np.allclose(w.Dp, tr["dir_from"], atol=5)
    avg = average_pings(r, verbose=False)
    assert avg.sizes["time"] == 9
    assert sorted(set(avg.n_pings.values.tolist())) == [180, 1024]
    # burst piu' lunghi del massimo (es. campionamento continuo): divisi in tratti consecutivi
    b2 = extract_wave_bursts(r, hab=0.5, max_samples=400, verbose=False)
    assert b2.sizes["sample"] == 400 and b2.n_samples.values.tolist() == [400, 400] * 3
    assert b2.attrs["n_short_bursts_skipped"] == 6 + 3            # i 6 da 180 ping e le 3 code da 224
    assert (np.diff(b2.time.values)[::2] == np.timedelta64(200, "s")).all()
    assert np.allclose(wave_spectra(b2).Hm0, 1.0, rtol=0.15)


def test_check_transform(sentinel, tmp_path):
    from adcp_pd0.diagnostics import check_transform
    r = StreamReader(sentinel["out"], pick_stream(sentinel["out"], "waves"))
    rows = check_transform(r, str(tmp_path), n_bursts=3, verbose=False)
    assert all(x["c_w4_deta"] > 0.9 for x in rows)          # w in fase con d(eta)/dt
    assert all(x["coh_w4"] > 0.9 and abs(x["ph_w4"]) < 20 for x in rows)
    assert all(x["coh_uh"] > 0.9 for x in rows)
    assert all(x["c_vb_deta"] > 0.9 for x in rows)
    assert (tmp_path / "check_transform.png").exists()
    # orientamento sbagliato -> correlazione negativa: il controllo se ne accorge
    rows_bad = check_transform(r, None, n_bursts=1, orientation="down", verbose=False)
    assert rows_bad[0]["c_w4_deta"] < -0.9


def test_archive_products(sentinel, tmp_path):
    """Metadati CF dei prodotti L2 e L3, tabelle e zip dei risultati."""
    md = json.load(open(os.path.join(REPO, "metadata_template.json")))
    md.update(latitude=44.4, longitude=8.9, station_id="TEST")
    mdf = tmp_path / "md.json"
    mdf.write_text(json.dumps(md))
    out = tmp_path / "o"
    pr = run_pre(sentinel["file"], "-o", out, "--metadata", mdf)
    assert pr["id"] == "TEST_20260710_20260710"
    pre, post = out / "preprocessing", out / "postprocessing"
    c = xr.open_dataset(pre / "TEST_20260710_20260710_L2_currents.nc")
    assert c.attrs["Conventions"].startswith("CF-1.8") and c.attrs["station_id"] == "TEST"
    assert c.attrs["data_level"] == "L2" and c.attrs["product"] == "currents"
    assert c.attrs["instrument_coordinates_recorded"] == "beam" and c.attrs["instrument_cell_size_m"] == 0.5
    assert "software_version" in c.attrs and c.attrs["proc_avg_bin_minutes"] == 60
    assert c.u.attrs["standard_name"] == "eastward_sea_water_velocity"
    assert c.temperature.attrs["standard_name"] == "sea_water_temperature"
    np.testing.assert_allclose(c.temperature.values, 15.0, atol=0.51)
    assert "deployment_qc" in c and "deployment_start" in c.attrs
    np.testing.assert_allclose(c.z.values[0], 1.68 + 0.6, atol=1e-4)     # quota sul fondale (hab predefinito 0.6 m)
    assert set(np.unique(c.vel_qc.values)) <= {1, 3, 4, 9}
    assert float(c.latitude) == 44.4
    b = xr.open_dataset(pre / "TEST_20260710_20260710_L2_wave_bursts.nc")
    assert b.attrs["product"] == "wave_bursts" and b.eta.attrs["units"] == "m"
    assert b.u.attrs["standard_name"] == "eastward_sea_water_velocity"
    np.testing.assert_allclose(b.level_height.values, b.level_range.values + 0.6, atol=1e-4)
    for name in ("deployment.json", "inventory.json", "metadata.json", "check_transform.json", "LEGGIMI_DATI.md"):
        assert (pre / name).exists(), name
    po = run_post("-o", out)
    w = xr.open_dataset(post / "TEST_20260710_20260710_L3_waves.nc")
    assert w.attrs["data_level"] == "L3" and w.attrs["product"] == "waves" and w.attrs["station_id"] == "TEST"
    assert w.Hm0.attrs["standard_name"] == "sea_surface_wave_significant_height"
    assert "flag_meanings" in w.wave_qc.attrs and "flag_masks" in w.direction_qc_cause.attrs
    g = w.wave_qc.values == 1
    assert g.sum() >= 5 and np.allclose(w.Hm0.values[g], sentinel["truth"]["Hs"], rtol=0.12)
    assert w.attrs["instrument_serial_number"] == "12345" and w.attrs["proc_nperseg"] == 256
    tabs = os.listdir(post / "tabelle")
    assert "onde_parametri.csv" in tabs and "serie_orarie.csv" in tabs
    df = pd.read_csv(post / "tabelle" / "onde_parametri.csv")
    assert {"Hm0", "Tp", "Dp", "wave_qc", "wave_qc_cause", "direction_qc"} <= set(df.columns)
    hh = pd.read_csv(post / "tabelle" / "serie_orarie.csv")
    assert {"temperatura", "pressure", "corrente_fondo_vel", "corrente_sup_dir", "deployment_qc"} <= set(hh.columns)
    # zip da inviare: tutto tranne le serie dei burst
    names = zipfile.ZipFile(po["zip"]).namelist()
    assert "postprocessing/risultati.json" in names
    assert "preprocessing/TEST_20260710_20260710_L2_currents.nc" in names
    assert "postprocessing/TEST_20260710_20260710_L3_waves.nc" in names
    assert any(n.endswith("_RAPPORTO_PREPROCESSING.md") for n in names)
    assert any(n.startswith("postprocessing/figure/") for n in names)
    assert any(n.startswith("preprocessing/figure_qc/") for n in names)
    assert not any(n.endswith("_L2_wave_bursts.nc") for n in names)
    res = json.load(open(post / "risultati.json", encoding="utf-8"))
    assert res["id"] == "TEST_20260710_20260710" and res["strumento"]["station_id"] == "TEST"
    assert abs(res["onde"]["Hm0"]["mediana"] - sentinel["truth"]["Hs"]) < 0.12


def test_cli_checks_and_reuse_conversion(sentinel, tmp_path, capsys):
    with pytest.raises(SystemExit):        # metadati mancanti: errore subito, niente output
        preprocess.main([sentinel["file"], "-o", str(tmp_path / "x"), "--metadata", "non_esiste.json"])
    assert not (tmp_path / "x").exists()
    with pytest.raises(SystemExit):        # post-processing senza pre-processing
        postprocess.main(["-o", str(tmp_path / "x")])
    with pytest.raises(SystemExit):        # nessun file e nessuna conversione
        preprocess.main(["-o", str(tmp_path / "y")])
    out = tmp_path / "o"
    run_pre(sentinel["file"], "-o", out)
    raw = sorted((out / "raw").rglob("*.nc"))
    mt = [f.stat().st_mtime_ns for f in raw]
    capsys.readouterr()
    run_pre(sentinel["file"], "-o", out)                 # stesso file: la conversione viene riusata
    assert "la riuso" in capsys.readouterr().out
    assert [f.stat().st_mtime_ns for f in raw] == mt
    run_pre("-o", out)                                   # senza file: usa la conversione presente
    assert [f.stat().st_mtime_ns for f in raw] == mt
    run_pre(sentinel["file"], "-o", out, "--riconverti")
    assert [f.stat().st_mtime_ns for f in sorted((out / "raw").rglob("*.nc"))] != mt


def test_motion_and_causes(tmp_path):
    """Telaio che ruota a meta' periodo: evento rilevato, flag 3 con causa 'movimento_telaio'."""
    from datetime import datetime
    from synthetic import make_sentinel_like
    parts = []
    f0 = tmp_path / "p.pd0"
    for h in range(10):
        hd, pt = (30.0, 2.0) if h < 5 else (38.0, 1.0)        # rotazione fra le 04:20 e le 05:00
        for start, n in ((0, 1024), (20, 180)):
            if start == 0 and h % 2:
                continue
            make_sentinel_like(str(f0), bursts=1, nping=n, depth=20.0, heading=hd, pitch=pt,
                               t0=datetime(2026, 7, 10, h, start), seed=10 * h + start)
            parts.append(f0.read_bytes())
    f = tmp_path / "mot.pd0"
    f.write_bytes(b"".join(parts))
    out = tmp_path / "o"
    run_pre(f, "-o", out)
    pre, post = out / "preprocessing", out / "postprocessing"
    dep = json.load(open(pre / "deployment.json"))
    assert len(dep["motion_events"]) == 1
    ev = dep["motion_events"][0]
    assert abs(ev["heading_before"] - 30) < 1 and abs(ev["heading_after"] - 38) < 1
    c = xr.open_dataset(one(pre, "*_L2_currents.nc"))
    mot = c.platform_motion.values.astype(bool)
    assert mot.sum() in (1, 2)
    q, cz = c.vel_qc.values, c.vel_qc_cause.values
    ok_cells = np.isfinite(c.u.values)
    assert (q[mot][ok_cells[mot]] == 3).all() and ((cz[mot] & 2) > 0).all()
    assert (q[~mot][ok_cells[~mot]] == 1).all()
    assert ((cz & 4) > 0).any()                              # cause diverse distinte (superficie)
    assert "flag_meanings" in c.vel_qc_cause.attrs
    b = xr.open_dataset(one(pre, "*_L2_wave_bursts.nc"))
    assert b.sizes["time"] == 5                              # solo i burst lunghi
    assert ((b.burst_qc.values == 3) == ((b.burst_qc_cause.values & 2) > 0)).all()
    rep = one(pre, "*_RAPPORTO_PREPROCESSING.md").read_text(encoding="utf-8")
    for txt in ("Periodo in posizione", "movimento_telaio", "Controllo della trasformazione",
                "## 6. Flag di qualita'", "fuori_posizione", "_L2_temperature.nc", "_L2_wave_bursts.nc"):
        assert txt in rep, txt
    assert "| 2026-07-10 0" in rep                       # riga della tabella dei movimenti
    assert (pre / "LEGGIMI_DATI.md").exists()
    run_post("-o", out)
    w = xr.open_dataset(one(post, "*_L3_waves.nc"))
    assert ((w.wave_qc.values == 3) == ((w.wave_qc_cause.values & 2) > 0)).all()
    assert (w.platform_motion.values == b.platform_motion.values).all()
    res = json.load(open(post / "risultati.json", encoding="utf-8"))
    assert len(res["movimenti_telaio"]) == 1 and abs(res["movimenti_telaio"][0]["rotazione_heading"] - 8) < 1
    mt = pd.read_csv(post / "tabelle" / "movimenti_telaio.csv")
    assert len(mt) == 1 and {"heading_prima", "heading_dopo", "Hm0_max_prima"} <= set(mt.columns)
    for fig in ("02_assetto_strumento.png", "31_temperatura_e_Hm0.png", "32_temperatura_e_corrente_fondo.png",
                "33_temperatura_e_corrente_superficie.png"):
        assert (post / "figure" / fig).exists(), fig
    assert "Movimenti del telaio" in one(post, "*_SINTESI_RISULTATI.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- analisi del post-processing
def _fake_currents(days=30, z0=12.0, seed=0, depth=34.5, hab=0.6):
    """Prodotto L2 delle correnti sintetico: due strati con il confine (taglio ed eco massimi) a z0."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-07-10 00:30", periods=24 * days, freq="1h")
    z = (1.68 + 0.5 * np.arange(64) + hab).astype(np.float32)
    nt, nz = len(t), len(z)
    s = np.tanh((z - z0) / 1.0)[None, :]                   # -1 sotto, +1 sopra il confine
    amp = (0.10 + 0.03 * np.sin(2 * np.pi * np.arange(nt) / (24 * 5.0)))[:, None]
    u = amp * s + rng.normal(0, 0.01, (nt, nz))
    v = 0.3 * amp * s + rng.normal(0, 0.01, (nt, nz))
    # eco: fondo lineare in dB + massimo a z0, riportato in counts (inverso della correzione usata)
    kc, alpha, noise = 0.45, 0.16, 40.0
    rs = (z - hab) / np.cos(np.deg2rad(25.0))
    sv = 70.0 - 0.3 * z + 6.0 * np.exp(-0.5 * ((z - z0) / 1.2) ** 2)
    ps = 10 ** ((sv - 20 * np.log10(rs) - 2 * alpha * rs) / 10.0)
    echo = (10.0 / kc) * np.log10(ps + 10 ** (kc * noise / 10.0))
    echo = echo[None, :] + rng.normal(0, 0.3, (nt, nz))
    side = z > depth * np.cos(np.deg2rad(25.0))            # celle vicine alla superficie: non valide
    qc = np.where(side[None, :], 4, 1).astype(np.int8) * np.ones((nt, 1), np.int8)
    u[:, side] = np.nan
    v[:, side] = np.nan
    c = xr.Dataset(coords=dict(time=("time", t.values), z=("z", z)))
    c["u"], c["v"] = (("time", "z"), u.astype(np.float32)), (("time", "z"), v.astype(np.float32))
    c["echo"] = (("time", "z"), echo.astype(np.float32))
    c["vel_qc"] = (("time", "z"), qc)
    c["deployment_qc"] = ("time", np.ones(nt, np.int8))
    c["water_depth"] = ("time", np.full(nt, depth, np.float32))
    c.attrs.update(instrument_height_above_bed_m=hab, instrument_beam_angle_deg=25.0)
    return c


@pytest.mark.parametrize("z0", [8.0, 14.0, 21.0])
def test_thermocline_known_interface(z0):
    from adcp_pd0 import thermocline as TH
    c = _fake_currents(days=20, z0=z0)
    df, extra = TH.thermocline(c)
    assert len(df) == 20 and extra["params"]["n_sensitivity_combinations"] == 48
    assert abs(df.z_taglio.median() - z0) <= 1.0 and df.affidabile.mean() > 0.9
    assert abs(df.z_eco.median() - z0) <= 1.5 and df.concordi.mean() > 0.9
    assert (df.z_taglio_p90 - df.z_taglio_p10).median() <= 2.0      # stima stabile al variare dei parametri
    assert np.isfinite(extra["S"]).any() and not df.al_fondo.any()
    ds = TH.to_dataset(df, dict(c.attrs), extra["params"])
    assert ds.attrs["data_level"] == "L3" and ds.z_shear.attrs["units"] == "m"
    assert ds.reliable.dtype == np.int8 and ds.sizes["time"] == 20
    sm = TH.summary(df)
    assert sm["n_intervalli"] == 20 and abs(sm["taglio_mediana_mensile"]["2026-07"] - z0) <= 1.0


def test_thermocline_descent_and_no_signal():
    """Confine che scende a meta' periodo; corrente uniforme = nessuna stima affidabile."""
    from adcp_pd0 import thermocline as TH
    a, b = _fake_currents(days=12, z0=18.0, seed=1), _fake_currents(days=12, z0=7.0, seed=2)
    b = b.assign_coords(time=b.time.values + np.timedelta64(12, "D"))
    df, _ = TH.thermocline(xr.concat([a, b], dim="time"))
    assert abs(df.z_taglio.iloc[:11].median() - 18.0) <= 1.0 and abs(df.z_taglio.iloc[13:].median() - 7.0) <= 1.0
    sm = TH.summary(df)
    assert sm["massima_discesa"]["variazione_m"] < -8
    flat = _fake_currents(days=10, z0=15.0, seed=3)
    rng = np.random.default_rng(4)
    flat["u"] = (("time", "z"), np.where(np.isfinite(flat.u.values), 0.1 + rng.normal(0, 0.01, flat.u.shape), np.nan))
    flat["v"] = (("time", "z"), np.where(np.isfinite(flat.v.values), rng.normal(0, 0.01, flat.v.shape), np.nan))
    df2, _ = TH.thermocline(flat, sensitivity=False)
    assert df2.affidabile.mean() < 0.5


def _fake_waves(hm, t0="2026-08-01", step_h=2, dp=200.0):
    t = pd.date_range(t0, periods=len(hm), freq=f"{step_h}h")
    n = len(hm)
    w = xr.Dataset(coords=dict(time=("time", t.values)))
    w["Hm0"] = ("time", np.asarray(hm, np.float32))
    w["Tp"] = ("time", np.full(n, 7.0, np.float32))
    w["Tm02"] = ("time", np.full(n, 5.0, np.float32))
    w["Dp"] = ("time", np.full(n, dp, np.float32))
    w["Ub"] = ("time", (0.1 * np.asarray(hm)).astype(np.float32))
    w["wave_qc"] = ("time", np.ones(n, np.int8))
    w["direction_qc"] = ("time", np.ones(n, np.int8))
    return w


def test_storms_and_wave_stats():
    from adcp_pd0 import stats as ST
    hm = np.full(240, 0.3)
    hm[20:26] = [1.1, 1.6, 2.2, 1.8, 1.2, 1.05]       # mareggiata 1 (12 h)
    hm[29] = 1.3                                      # a 8 h dalla precedente: stessa mareggiata
    hm[100:103] = [1.2, 1.4, 1.1]                     # mareggiata 2
    hm[200] = np.nan
    w = _fake_waves(hm)
    w["wave_qc"].values[101] = 4                      # un burst non valido non conta
    st = ST.storms(w, thr=1.0, sep_hours=12.0)
    assert len(st) == 2
    assert st.Hm0_max.round(2).tolist() == [2.2, 1.2] and st.n_burst.tolist() == [7, 2]
    assert st.inizio.iloc[0] == pd.Timestamp("2026-08-02 16:00") and st.durata_h.iloc[0] == 20.0
    assert len(ST.storms(w, thr=3.0)) == 0
    dep = dict(motion_events=[dict(start="2026-08-02 21:00", end="2026-08-02 22:30", heading_before=320.0,
                                   heading_after=329.5)])
    T = pd.Series(18.0, index=pd.date_range("2026-08-01", periods=480, freq="1h"))
    T[T.index > pd.Timestamp("2026-08-03 12:00")] = 20.0
    st2 = ST.enrich_storms(st, None, T, dep)
    assert st2.movimento_telaio.tolist() == [True, False] and abs(st2.rotazione_heading.iloc[0] - 9.5) < 1e-6
    assert st2.dT_fondo.iloc[0] > 1.0 and abs(st2.dT_fondo.iloc[1]) < 1e-6
    mt = ST.motion_table(dep, w)
    assert len(mt) == 1 and abs(mt.Hm0_max_prima.iloc[0] - 2.2) < 1e-6
    ws = ST.wave_stats(w)
    assert ws["n_validi"] == 238 and ws["Hm0"]["max"] == 2.2 and ws["Hm0"]["quando_max"] == "2026-08-02 20:00:00"
    assert ws["direzione"]["settore_prevalente"] == "SSO" and ws["superamenti_pct"][">1 m"] > 0
    jt = ST.joint_table(w.Dp.values, w.Hm0.values, [0, 0.5, 1.0, 1.5, np.inf], "m")
    assert abs(jt.loc["totale", "totale"] - 100.0) < 1e-6 and abs(jt.loc["SSO", "totale"] - 100.0) < 1e-6
    assert jt.attrs["n"] == 239
    mo = ST.monthly_table(w, None, T, None)
    assert list(mo.index) == ["2026-08"] and abs(mo.Hm0_max.iloc[0] - 2.2) < 1e-6


def test_tide_rotary_and_temperature():
    from adcp_pd0 import stats as ST
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-07-10 00:30", periods=24 * 60, freq="1h")
    th = np.arange(len(t), dtype=float)
    lev = 0.08 * np.cos(2 * np.pi * th / 12.4206 + 1.0) + 0.03 * np.cos(2 * np.pi * th / 23.9345 - 0.5) \
        + rng.normal(0, 0.01, len(t))
    lev[500:520] = np.nan                              # un buco nei dati
    lev[-100:] = np.nan                                # fine delle misure (strumento recuperato)
    tide = ST.tide_fit(t, lev)
    cons = tide["info"]["costituenti"]
    assert set(cons) == {"M2", "S2", "N2", "K1", "O1"}
    assert abs(cons["M2"]["ampiezza_cm"] - 8.0) < 0.3 and abs(cons["K1"]["ampiezza_cm"] - 3.0) < 0.3
    assert cons["S2"]["ampiezza_cm"] < 0.5 and tide["info"]["varianza_spiegata"] > 0.95
    assert tide["marea"].iloc[-50:].isna().all() and tide["marea"].iloc[505:515].notna().all()
    assert set(ST.tide_fit(t[:24 * 10], lev[:24 * 10])["info"]["costituenti"]) == {"M2", "K1"}   # serie corta
    assert ST.tide_fit(t[:30], lev[:30]) is None
    # corrente che ruota in senso orario con periodo 17 h: energia nella componente oraria
    f0 = 24.0 / 17.0
    u = 0.1 * np.cos(2 * np.pi * f0 * th / 24.0) + rng.normal(0, 0.005, len(t))
    v = -0.1 * np.sin(2 * np.pi * f0 * th / 24.0) + rng.normal(0, 0.005, len(t))
    sp = ST.rotary_spectrum(u, v, dt_hours=1.0)
    i = int(np.argmax(sp["orario"]))
    assert abs(sp["f"][i] - f0) < 0.1 and sp["orario"][i] > 100 * sp["antiorario"][i]
    bands = ST.band_variance(sp, latitude=44.4)
    assert abs(bands["periodo_inerziale_h"] - 17.1) < 0.1 and bands["inerziale"] > 0.8
    assert bands["inerziale_quota_oraria"] > 0.95
    pa = ST.principal_axis(u, 0.5 * u)
    assert abs(pa["direzione_gradi"] - np.degrees(np.arctan2(1, 0.5))) < 0.5 and pa["frazione_varianza"] > 0.99
    x, y, frac = ST.progressive_vector(np.full(25, 0.1), np.full(25, -0.2), t[:25])
    assert abs(x[-1] - 0.1 * 25 * 3.6) < 0.2 and abs(y[-1] + 0.2 * 25 * 3.6) < 0.4 and frac == 1.0
    # temperatura: salto di 4 gradi con oscillazioni di 12.4 h attorno al salto
    T = pd.Series(18.0 + 0.1 * rng.standard_normal(len(t)), index=t)
    T[t >= pd.Timestamp("2026-08-10")] += 4.0
    osc = (t >= pd.Timestamp("2026-08-06")) & (t < pd.Timestamp("2026-08-18"))
    T[osc] += 1.0 * np.sin(2 * np.pi * th[osc] / 12.42)
    z = ST.max_change_window(T.resample("1D").mean())
    assert abs(z["delta"] - 4.0) < 0.5 and z["da"] <= pd.Timestamp("2026-08-10") <= z["a"]
    per = ST.dominant_periods(T[z["t0"]:z["t1"]])
    assert abs(per[0]["periodo_h"] - 12.4) < 0.4
    ts = ST.temperature_stats(T)
    assert abs(ts["massima_variazione"]["delta"] - 4.0) < 0.5 and ts["n_ore"] == len(t)


def test_layers_and_qc_summary():
    from adcp_pd0 import stats as ST
    c = _fake_currents(days=3, z0=15.0)
    c["deployment_qc"].values[-5:] = 4                     # ultime ore fuori posizione
    c["vel_qc_cause"] = (("time", "z"), np.where(c.vel_qc.values == 4, 4, 0).astype(np.uint16))
    c["deployment_qc_cause"] = ("time", np.where(c.deployment_qc.values == 4, 1, 0).astype(np.uint16))
    L = ST.layer_series(c)
    assert L.vel_fondo.iloc[-5:].isna().all() and L.vel_fondo.iloc[:-5].notna().all()
    assert (L.u_fondo.iloc[:-5] < 0).all() and (L.u_sup.iloc[:-5] > 0).all()       # strati opposti
    assert L.attrs["fondo"].startswith("2.3-3.3 m") and "sotto la superficie" in L.attrs["sup"]
    assert abs(L.attrs["prof_sup"] - 4.0) < 1.5
    st = ST.speed_stats(L.u_sup.values, L.v_sup.values, L.index)
    assert st["settore_prevalente"] == "ENE" and st["n"] == len(L) - 5
    q = ST.qc_summary(c)
    assert q["velocita"]["per_causa"]["contaminazione_superficie"] > 0
    assert q["posizione"]["per_flag"][4] == 5 and q["velocita"]["celle_utili_buone_pct"] == 100.0


def test_motion_events_merge_and_net_rotation():
    """Con un burst all'ora due salti in ore consecutive sono un solo episodio; la rotazione di una
    mareggiata e' quella netta, non la somma dei salti."""
    from adcp_pd0.deployment import motion_events
    from adcp_pd0 import stats as ST
    # minuti con dati: tre minuti per ora (HH:20-HH:22) per tre giorni, come i burst brevi del file reale
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-20") + pd.Timedelta(hours=h, minutes=20 + k)
                            for h in range(72) for k in range(3)])
    hour = ((idx - idx[0]).total_seconds().values // 3600).astype(int)
    # salti fra le 22 e le 23 e fra le 00 e le 01 (a 61 minuti l'uno dall'altro), poi un terzo il 22 agosto
    head = np.select([hour <= 22, hour <= 24, hour <= 60], [319.5, 324.0, 329.0], default=331.5)
    m = pd.DataFrame(dict(heading=head, pitch=1.0, roll=-3.0), index=idx)
    end = idx[-1] + pd.Timedelta(minutes=1)
    ev = motion_events(m, idx[0], end, ctx=200)
    assert len(ev) == 2
    assert ev[0]["start"] == "2026-08-20 22:22:00" and ev[0]["end"] == "2026-08-21 01:21:00"
    # assetto prima e dopo senza i dati dell'altro episodio, anche con una finestra molto lunga
    assert abs(ev[0]["heading_before"] - 319.5) < 1e-6 and abs(ev[0]["heading_after"] - 329.0) < 1e-6
    assert abs(ev[1]["heading_before"] - 329.0) < 1e-6 and abs(ev[1]["heading_after"] - 331.5) < 1e-6
    ev60 = motion_events(m, idx[0], end, merge_min=60.0, ctx=200)
    assert len(ev60) == 3 and abs(ev60[0]["heading_after"] - 324.0) < 1e-6
    # mareggiata dalle 18:00 del 20 alle 02:00 del 21: rotazione netta 329.0 - 319.5
    hm = np.full(36, 0.3)
    hm[9:14] = [1.2, 1.8, 1.9, 1.5, 1.1]
    w = _fake_waves(hm, t0="2026-08-20")
    for events in (ev, ev60):
        st = ST.enrich_storms(ST.storms(w), None, None, dict(motion_events=events))
        assert len(st) == 1 and bool(st.movimento_telaio.iloc[0])
        assert abs(st.rotazione_heading.iloc[0] - 9.5) < 1e-6
    mt = ST.motion_table(dict(motion_events=ev), w)
    assert len(mt) == 2 and abs(mt.rotazione_heading.iloc[0] - 9.5) < 1e-6
    assert abs(mt.Hm0_max_prima.iloc[0] - 1.9) < 1e-6
