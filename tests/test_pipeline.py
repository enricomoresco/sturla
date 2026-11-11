import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic import make_file  # noqa: E402
from adcp_pd0 import convert_files, StreamReader, pick_stream, qc_currents, analyze_waves  # noqa: E402
from adcp_pd0.pd0 import index_file  # noqa: E402
from adcp_pd0.transform import beam_to_inst, inst_to_earth  # noqa: E402
import process_pd0  # noqa: E402


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
    name = pick_stream(data["out"], "waves")
    r = StreamReader(data["out"], name)
    w = analyze_waves(r, hab=data["truth"]["hab"], puv_cell=2, verbose=False)
    assert w.sizes["time"] == data["truth"]["hours"]
    assert np.allclose(w.Hm0, data["truth"]["Hs"], rtol=0.08)
    assert np.allclose(w.Tp, data["truth"]["Tp"], rtol=0.06)
    assert np.allclose(w.Hm0_vb, data["truth"]["Hs"], rtol=0.15)
    assert np.allclose(w.Dp, data["truth"]["dir_from"], atol=5)
    assert np.allclose(w.depth, data["truth"]["depth"], atol=0.05)


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


def test_cli_all(data, tmp_path):
    out = tmp_path / "cli"
    process_pd0.main(["all", data["file"], "-o", str(out), "--hab", "0.5", "--puv-cell", "2"])
    figs = os.listdir(out / "figures")
    for f in ("currents_sections.png", "currents_summary.png", "waves_bulk.png",
              "waves_spectrogram.png", "waves_burst_example.png"):
        assert f in figs
    assert (out / "currents_qc.nc").exists() and (out / "waves_bulk.nc").exists()


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
    tr = sentinel["truth"]
    r = StreamReader(sentinel["out"], pick_stream(sentinel["out"], "waves"))
    w = analyze_waves(r, hab=tr["hab"], puv_cell=20, verbose=False)
    assert np.allclose(w.Hm0_vb, tr["Hs"], rtol=0.12)
    assert np.allclose(w.Tp_vb, tr["Tp"], rtol=0.06)
    assert np.allclose(w.Dp, tr["dir_from"], atol=5)
    assert (w.dir_ref_vb == 1).all()


def test_sentinel_like_cli(sentinel, tmp_path):
    out = tmp_path / "cli"
    process_pd0.main(["all", sentinel["file"], "-o", str(out), "--puv-cell", "20"])
    assert (out / "currents_qc.nc").exists() and (out / "currents_avg.nc").exists()
    assert (out / "figures" / "currents_sections.png").exists()


def test_trash_and_range(tmp_path):
    import json
    import xarray as xr
    from synthetic import make_trashy
    from adcp_pd0.pd0 import parse_byte_range
    f = tmp_path / "trash.pd0"
    tr = make_trashy(str(f), bursts=6, depth=20.0)
    size = os.path.getsize(f)
    assert parse_byte_range("25%:75%", size) == (size // 4, int(size * 0.75))
    assert parse_byte_range(":1", size) == (0, 2**20)
    # 1) file intero: periodo in posizione trovato in automatico, il resto con flag 4
    out = tmp_path / "all"
    process_pd0.main(["all", str(f), "-o", str(out), "--puv-cell", "auto"])
    dep = json.load(open(out / "deployment.json"))
    t0, t1 = np.datetime64(dep["start"]), np.datetime64(dep["end"])
    assert np.datetime64(tr["t0"]) <= t0 <= np.datetime64(tr["t0"]) + np.timedelta64(30, "m")
    last = np.datetime64(tr["t_end"]) - np.timedelta64(30, "m") + np.timedelta64(512, "s")   # fine ultimo burst buono
    assert last - np.timedelta64(15, "m") <= t1 <= last
    assert (out / "figures" / "deployment.png").exists()
    arch = out / "archive"
    c = xr.open_dataset(next(arch.glob("*_currents.nc")))
    w = xr.open_dataset(next(arch.glob("*_waves.nc")))
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
    assert w.wave_qc.values.tolist().count(1) >= 4
    assert (w.wave_qc.values[w.wave_qc.values != 1] == 4).all()
    good = w.wave_qc.values == 1
    assert np.allclose(w.Hm0.values[good], tr["Hs"], rtol=0.12)
    # temperatura oraria: in posizione 14.5-15.5 °C (flag 1), in aria 25 °C (flag 4)
    th = xr.open_dataset(next(arch.glob("*_temperature_hourly.nc")))
    q = th.temperature_qc.values
    assert (q == 1).sum() >= 2 and (q == 4).sum() >= 2
    assert np.all(np.abs(th.temperature.values[q == 1] - 15.0) <= 0.51)
    assert np.all(th.temperature.values[q == 4] > 14.4)
    assert th.temperature.attrs["cell_methods"].startswith("time: mean")
    assert "time_bnds" in th
    assert next(arch.glob("*_temperature_hourly.csv")).exists()
    assert (out / "figures" / "temperature_hourly.png").exists()
    # 2) con --trim i dati fuori posizione spariscono
    out3 = tmp_path / "trim"
    process_pd0.main(["all", str(f), "-o", str(out3), "--trim"])
    c3 = xr.open_dataset(next((out3 / "archive").glob("*_currents.nc")))
    assert (c3.deployment_qc.values == 1).all()
    # 3) solo la metà centrale
    out2 = tmp_path / "mid"
    process_pd0.main(["all", str(f), "-o", str(out2), "--range", "25%:75%", "--puv-cell", "auto"])
    w2 = xr.load_dataset(out2 / "waves_bulk.nc")
    ok2 = np.isfinite(w2.Hm0_vb.values)
    assert ok2.sum() >= 1 and np.allclose(w2.Hm0_vb.values[ok2], tr["Hs"], rtol=0.12)


def test_mixed_bursts(tmp_path):
    """Come il file reale: burst da 1024 ping (onde) alternati a burst da 180 ping."""
    from synthetic import make_sentinel_like
    from adcp_pd0.currents import average_pings
    f = tmp_path / "mix.pd0"
    tr = make_sentinel_like(str(f), bursts=9, interval_min=40, nping=[1024, 180, 180], depth=34.5)
    out = tmp_path / "out"
    convert_files([str(f)], str(out), verbose=False)
    r = StreamReader(str(out), pick_stream(str(out), "waves"))
    w = analyze_waves(r, hab=0.5, puv_cell="auto", verbose=False)
    assert w.sizes["time"] == 3 and w.attrs["n_short_bursts_skipped"] == 6
    assert np.allclose(w.Hm0_vb, 1.0, rtol=0.12) and np.allclose(w.Dp, tr["dir_from"], atol=5)
    avg = average_pings(r, verbose=False)
    assert avg.sizes["time"] == 9
    assert sorted(set(avg.n_pings.values.tolist())) == [180, 1024]


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


def test_archive(sentinel, tmp_path):
    import json
    import xarray as xr
    md = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "metadata_template.json")))
    md.update(latitude=44.4, longitude=8.9, station_id="TEST")
    mdf = tmp_path / "md.json"
    mdf.write_text(json.dumps(md))
    out = tmp_path / "o"
    process_pd0.main(["all", sentinel["file"], "-o", str(out), "--puv-cell", "auto", "--metadata", str(mdf)])
    files = os.listdir(out / "archive")
    cur = [f for f in files if f.endswith("_currents.nc")][0]
    wav = [f for f in files if f.endswith("_waves.nc")][0]
    c = xr.open_dataset(out / "archive" / cur)
    w = xr.open_dataset(out / "archive" / wav)
    assert c.attrs["Conventions"].startswith("CF-1.8") and c.attrs["station_id"] == "TEST"
    assert c.u.attrs["standard_name"] == "eastward_sea_water_velocity"
    assert c.temperature.attrs["standard_name"] == "sea_water_temperature"
    np.testing.assert_allclose(c.temperature.values, 15.0, atol=0.51)
    assert "deployment_qc" in c and "deployment_start" in c.attrs
    np.testing.assert_allclose(c.z.values[0], 1.68 + 0.5, atol=1e-4)     # quota sul fondale
    assert set(np.unique(c.vel_qc.values)) <= {1, 3, 4, 9}
    assert float(c.latitude) == 44.4
    assert w.Hm0.attrs["standard_name"] == "sea_surface_wave_significant_height"
    g = w.wave_qc.values == 1
    assert g.sum() >= 5 and np.allclose(w.Hm0.values[g], sentinel["truth"]["Hs"], rtol=0.12)
    assert w.attrs["instrument_serial_number"] == "12345"
    assert any(f.endswith("_waves.csv") for f in files) and any(f.endswith("_timeseries.csv") for f in files)


def test_cli_checks_and_skip_convert(sentinel, tmp_path):
    with pytest.raises(SystemExit):        # metadati mancanti: errore subito
        process_pd0.main(["all", sentinel["file"], "-o", str(tmp_path / "x"), "--metadata", "non_esiste.json"])
    assert not (tmp_path / "x").exists()
    out = tmp_path / "o"
    process_pd0.main(["convert", sentinel["file"], "-o", str(out)])
    process_pd0.main(["all", "--skip-convert", "-o", str(out)])
    assert any(f.endswith("_waves.nc") for f in os.listdir(out / "archive"))


def test_motion_and_causes(tmp_path):
    """Telaio che ruota a meta' periodo: evento rilevato, flag 3 con causa 'movimento_telaio'."""
    import json
    import xarray as xr
    from datetime import datetime, timedelta
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
    process_pd0.main(["all", str(f), "-o", str(out), "--puv-cell", "auto"])
    dep = json.load(open(out / "deployment.json"))
    assert len(dep["motion_events"]) == 1
    ev = dep["motion_events"][0]
    assert abs(ev["heading_before"] - 30) < 1 and abs(ev["heading_after"] - 38) < 1
    arch = out / "archive"
    c = xr.open_dataset(next(arch.glob("*_currents.nc")))
    mot = c.platform_motion.values.astype(bool)
    assert mot.sum() in (1, 2)
    q, cz = c.vel_qc.values, c.vel_qc_cause.values
    ok_cells = np.isfinite(c.u.values)
    assert (q[mot][ok_cells[mot]] == 3).all() and ((cz[mot] & 2) > 0).all()
    assert (q[~mot][ok_cells[~mot]] == 1).all()
    assert ((cz & 4) > 0).any()                              # cause diverse distinte (superficie)
    assert "flag_meanings" in c.vel_qc_cause.attrs
    w = xr.open_dataset(next(arch.glob("*_waves.nc")))
    assert ((w.wave_qc.values == 3) == ((w.wave_qc_cause.values & 2) > 0)).all()
    import pandas as pd
    df = pd.read_csv(next(arch.glob("*_timeseries.csv")), comment="#")
    assert "deployment_qc_cause" in df and "platform_motion" in df
    rep = next(arch.glob("*_RAPPORTO_ELABORAZIONE.md")).read_text(encoding="utf-8")
    for txt in ("Periodo in posizione", "movimento_telaio", "Controllo della trasformazione",
                "## 6. Flag di qualita'", "fuori_posizione", "_temperature_hourly.nc"):
        assert txt in rep, txt
    assert "| 2026-07-10 0" in rep                       # riga della tabella dei movimenti
    assert (arch / "LEGGIMI_DATI.md").exists()
