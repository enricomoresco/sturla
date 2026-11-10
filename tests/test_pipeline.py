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
