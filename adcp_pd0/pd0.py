"""Lettore PD0 (Teledyne RDI: Sentinel V, Workhorse, ...) veloce e a memoria limitata.

Strategia per file grandi (GB):
1. ``index_file``: scansione sequenziale del file mappato in memoria (mmap), senza
   caricarlo in RAM. Per ogni ensemble si registra solo posizione e "layout"
   (numero di byte + elenco degli ID/offset dei data type). Se la struttura si rompe
   (byte spuri, file troncato) si risincronizza cercando il prossimo header 0x7F7F.
2. ``iter_decode``: gli ensemble con lo stesso layout hanno i campi agli stessi
   offset, quindi vengono decodificati in blocco con numpy (a pezzi di ~150 MB),
   verificando il checksum di tutti gli ensemble in modo vettoriale.
3. Gli ensemble sono poi separati in "stream" in base alla configurazione
   (n. celle, dimensione cella, ping per ensemble, ...): tipicamente uno stream
   per le correnti (profili mediati) e uno per le onde (burst di ping singoli).

Riferimento formato: Teledyne RDI "Workhorse Commands and Output Data Format" e
manuale Sentinel V (data type del 5° beam verticale).
"""
from __future__ import annotations

import mmap
import os
import warnings
from array import array
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HEADER = b"\x7f\x7f"

# ID dei data type. Gli ID con bit 0 = 1 (0x0001, 0x0081, ...) sono usati per il
# secondo profilo nelle modalità a doppia configurazione: trattati come i primi.
ID_NAMES = {
    0x0000: "fixed_leader", 0x0001: "fixed_leader",
    0x0080: "variable_leader", 0x0081: "variable_leader",
    0x0100: "velocity", 0x0101: "velocity",
    0x0200: "correlation", 0x0201: "correlation",
    0x0300: "echo", 0x0301: "echo",
    0x0400: "percent_good", 0x0401: "percent_good",
    0x0500: "status",
    0x0600: "bottom_track",
    0x0A00: "vb_velocity",
    0x0B00: "vb_correlation",
    0x0C00: "vb_echo",
    0x0D00: "vb_percent_good",
    0x0F01: "vb_leader",
    0x3200: "transformation_matrix",
}

COORD_NAMES = {0: "beam", 1: "instrument", 2: "ship", 3: "earth"}


def id_name(i: int) -> str:
    return ID_NAMES.get(i, f"unknown_0x{i:04X}")


@dataclass
class Layout:
    """Struttura di un ensemble: lunghezza e posizione dei data type."""
    nbytes: int                 # byte dell'ensemble, checksum esclusa
    ids: tuple                  # ID dei data type
    offsets: tuple              # offset (dall'inizio dell'ensemble)
    count: int = 0

    @property
    def sizes(self) -> tuple:
        ends = list(self.offsets[1:]) + [self.nbytes]
        return tuple(e - o for o, e in zip(self.offsets, ends))

    @property
    def valid(self) -> bool:
        hl = 6 + 2 * len(self.ids)
        o = self.offsets
        return (len(o) > 0 and o[0] >= hl and all(b > a for a, b in zip(o, o[1:]))
                and o[-1] < self.nbytes)

    def find(self, name: str):
        """(offset, size) del primo data type con quel nome, oppure None."""
        for i, o, s in zip(self.ids, self.offsets, self.sizes):
            if id_name(i) == name:
                return o, s
        return None

    def names(self) -> list:
        return [id_name(i) for i in self.ids]


@dataclass
class FileIndex:
    path: str
    size: int
    positions: np.ndarray       # posizione (byte) di ogni ensemble
    layout_id: np.ndarray       # indice in ``layouts``
    layouts: list
    n_resync: int = 0           # quante volte si è dovuto risincronizzare
    stats: dict = field(default_factory=dict)


def _checksum_ok(mm, pos: int, nb: int) -> bool:
    if pos + nb + 2 > len(mm):
        return False
    s = int(np.frombuffer(mm, np.uint8, count=nb, offset=pos).sum(dtype=np.uint64)) & 0xFFFF
    return s == (mm[pos + nb] | (mm[pos + nb + 1] << 8))


def index_file(path: str, max_bytes: int | None = None, verbose: bool = True) -> FileIndex:
    """Indicizza un file PD0 senza leggerlo in memoria.

    ``max_bytes`` limita la scansione ai primi N byte (utile per un'ispezione veloce).
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    end = size if max_bytes is None else min(size, int(max_bytes))
    positions = array("q")
    lay_ids = array("i")
    layout_keys: dict = {}
    n_resync = 0
    report_every = 256 * 2**20
    next_report = report_every

    find = mm.find
    pos = find(HEADER, 0, end)
    while 0 <= pos <= end - 6:
        nb = mm[pos + 2] | (mm[pos + 3] << 8)
        ndt = mm[pos + 4 + 1]
        hl = 6 + 2 * ndt
        nxt = pos + nb + 2
        ok = 0 < ndt < 64 and nb >= hl and nxt <= size
        if ok:
            # controllo rapido: dopo l'ensemble deve iniziare un altro header;
            # altrimenti (fine file, byte spuri) si verifica la checksum.
            if nxt + 1 < size and mm[nxt] == 0x7F and mm[nxt + 1] == 0x7F:
                pass
            else:
                ok = _checksum_ok(mm, pos, nb)
        if ok:
            key = mm[pos + 2:pos + hl]
            li = layout_keys.get(key)
            if li is None:
                li = layout_keys[key] = len(layout_keys)
            positions.append(pos)
            lay_ids.append(li)
            pos = nxt
            if pos >= end - 6:
                break
            if not (mm[pos] == 0x7F and mm[pos + 1] == 0x7F):
                n_resync += 1
                pos = find(HEADER, pos, end)
        else:
            n_resync += 1
            pos = find(HEADER, pos + 1, end)
        if verbose and pos > next_report:
            print(f"  indicizzati {pos / 2**20:,.0f} / {end / 2**20:,.0f} MB", flush=True)
            next_report += report_every

    layouts = []
    for key in layout_keys:  # dict mantiene l'ordine di inserimento
        nb = key[0] | (key[1] << 8)
        ndt = key[3]
        offs = np.frombuffer(key[4:4 + 2 * ndt], "<u2")
        # gli ID si leggono dal primo ensemble con quel layout (vedi sotto)
        layouts.append(Layout(nbytes=nb, ids=(), offsets=tuple(int(o) for o in offs)))
    positions = np.frombuffer(positions, np.int64).copy() if len(positions) else np.zeros(0, np.int64)
    lay_ids = np.frombuffer(lay_ids, np.int32).copy() if len(lay_ids) else np.zeros(0, np.int32)
    for li, lay in enumerate(layouts):
        sel = lay_ids == li
        lay.count = int(sel.sum())
        p0 = int(positions[np.argmax(sel)])
        lay.ids = tuple(mm[p0 + o] | (mm[p0 + o + 1] << 8) for o in lay.offsets if o + 1 < lay.nbytes)
        if len(lay.ids) != len(lay.offsets):
            lay.ids = lay.ids + (0xFFFF,) * (len(lay.offsets) - len(lay.ids))
    covered = int(sum((l.nbytes + 2) * l.count for l in layouts))
    idx = FileIndex(path=path, size=size, positions=positions, layout_id=lay_ids,
                    layouts=layouts, n_resync=n_resync,
                    stats=dict(bytes_scanned=end, bytes_in_ensembles=covered,
                               bytes_skipped=end - covered, n_ensembles=len(positions)))
    idx._mm = mm  # tenuto aperto per la decodifica
    return idx


# --------------------------------------------------------------------------
# decodifica vettoriale
# --------------------------------------------------------------------------
def _v(M: np.ndarray, o: int, dt: str) -> np.ndarray:
    n = np.dtype(dt).itemsize
    return np.ascontiguousarray(M[:, o:o + n]).view(dt)[:, 0]


def _decode_fixed(M, o, s) -> dict:
    g = lambda off, dt: _v(M, o + off, dt)
    d = dict(
        fw_ver=g(2, "u1"), fw_rev=g(3, "u1"), sysconfig=g(4, "<u2"),
        n_beams=g(8, "u1"), n_cells=g(9, "u1"), pings_per_ens=g(10, "<u2"),
        cell_size_cm=g(12, "<u2"), blank_cm=g(14, "<u2"), prof_mode=g(16, "u1"),
        corr_threshold=g(17, "u1"), pg_min=g(19, "u1"), err_vel_max_mm=g(20, "<u2"),
        tpp_hund=(g(22, "u1").astype(np.int32) * 6000 + g(23, "u1").astype(np.int32) * 100
                  + g(24, "u1")),
        coord_xfrm=g(25, "u1"), heading_align=g(26, "<i2"), heading_bias=g(28, "<i2"),
        sensor_source=g(30, "u1"), sensors_avail=g(31, "u1"), bin1_dist_cm=g(32, "<u2"),
        xmit_pulse_cm=g(34, "<u2"),
    )
    if s >= 58:
        d["serial"] = g(54, "<u4")
    if s >= 59:
        d["beam_angle"] = g(58, "u1")
    return d


def _decode_variable(M, o, s) -> dict:
    g = lambda off, dt: _v(M, o + off, dt)
    d = dict(
        ens_num=g(2, "<u2").astype(np.int64) + g(11, "u1").astype(np.int64) * 65536,
        bit=g(12, "<u2"),
        sound_speed=g(14, "<u2").astype(np.float32),
        depth_xducer=g(16, "<u2").astype(np.float32) / 10.0,
        heading=g(18, "<u2").astype(np.float32) / 100.0,
        pitch=g(20, "<i2").astype(np.float32) / 100.0,
        roll=g(22, "<i2").astype(np.float32) / 100.0,
        salinity=g(24, "<u2").astype(np.float32),
        temperature=g(26, "<i2").astype(np.float32) / 100.0,
        heading_std=g(31, "u1").astype(np.float32),
        pitch_std=g(32, "u1").astype(np.float32) / 10.0,
        roll_std=g(33, "u1").astype(np.float32) / 10.0,
    )
    if s >= 46:
        d["error_status"] = g(42, "<u4")
    if s >= 56:
        # deca-pascal -> dbar
        d["pressure"] = g(48, "<u4").astype(np.float64) * 1e-3
        d["pressure_var"] = g(52, "<u4").astype(np.float64) * 1e-3
    # tempo: orologio Y2K (byte 57-64) se presente, altrimenti RTC a 2 cifre
    year = g(4, "u1").astype(np.int32) + 2000
    mo, dd, hh, mi, ss, hu = (g(k, "u1").astype(np.int32) for k in range(5, 11))
    if s >= 65:
        cen = g(57, "u1").astype(np.int32)
        ok = (cen >= 19) & (cen <= 21)
        if ok.any():
            year = np.where(ok, cen * 100 + g(58, "u1").astype(np.int32), year)
            mo2, dd2, hh2, mi2, ss2, hu2 = (g(k, "u1").astype(np.int32) for k in range(59, 65))
            mo, dd, hh = np.where(ok, mo2, mo), np.where(ok, dd2, dd), np.where(ok, hh2, hh)
            mi, ss, hu = np.where(ok, mi2, mi), np.where(ok, ss2, ss), np.where(ok, hu2, hu)
    t = pd.to_datetime(pd.DataFrame(dict(year=year, month=mo, day=dd, hour=hh,
                                         minute=mi, second=ss)), errors="coerce")
    d["time"] = (t + pd.to_timedelta(hu * 10, unit="ms")).to_numpy("datetime64[ns]")
    return d


def _block(M, o, s, n_expected, dt):
    """Array (N, n) di un data type 'ID + n valori'."""
    isz = np.dtype(dt).itemsize
    n = min(n_expected, (s - 2) // isz)
    return np.ascontiguousarray(M[:, o + 2:o + 2 + n * isz]).view(dt), n


def decode_matrix(M: np.ndarray, lay: Layout) -> dict:
    """Decodifica una matrice (N ensemble x nbytes) con layout uniforme."""
    out = {}
    fl = lay.find("fixed_leader")
    vl = lay.find("variable_leader")
    if fl is None or vl is None:
        raise ValueError("ensemble senza fixed/variable leader")
    out.update(_decode_fixed(M, *fl))
    out.update(_decode_variable(M, *vl))
    nc = int(out["n_cells"][0])
    nbm = int(out["n_beams"][0])
    vel = lay.find("velocity")
    if vel is not None and nc > 0:
        nb_eff = min(nbm, (vel[1] - 2) // (2 * nc))
        a, _ = _block(M, *vel, nc * nb_eff, "<i2")
        v = a.reshape(len(M), nc, nb_eff).astype(np.float32)
        v[a.reshape(v.shape) == -32768] = np.nan
        out["vel"] = v / 1000.0
        for name, key in (("correlation", "corr"), ("echo", "echo"), ("percent_good", "pg")):
            b = lay.find(name)
            if b is not None:
                arr, n = _block(M, *b, nc * nb_eff, "u1")
                if n == nc * nb_eff:
                    out[key] = arr.reshape(len(M), nc, nb_eff)
    vbl = lay.find("vb_leader")
    vbv = lay.find("vb_velocity")
    nvb = None
    if vbv is not None:
        nvb = (vbv[1] - 2) // 2
    elif lay.find("vb_echo") is not None:
        nvb = lay.find("vb_echo")[1] - 2
    if nvb:
        if vbl is not None:
            g = lambda off, dt: _v(M, vbl[0] + off, dt)
            out["vb_n_cells"] = g(2, "<u2")
            out["vb_pings"] = g(4, "<u2")
            out["vb_cell_size_cm"] = g(6, "<u2")
            out["vb_first_cell_cm"] = g(8, "<u2")
            nlead = int(out["vb_n_cells"][0])
            if 0 < nlead <= nvb:
                nvb = nlead
        if vbv is not None:
            a, _ = _block(M, *vbv, nvb, "<i2")
            v = a.astype(np.float32)
            v[a == -32768] = np.nan
            out["vb_vel"] = v / 1000.0
        for name, key in (("vb_correlation", "vb_corr"), ("vb_echo", "vb_echo"),
                          ("vb_percent_good", "vb_pg")):
            b = lay.find(name)
            if b is not None:
                arr, n = _block(M, *b, nvb, "u1")
                if n == nvb:
                    out[key] = arr
    return out


CONFIG_KEYS = ("n_cells", "n_beams", "pings_per_ens", "cell_size_cm", "blank_cm",
               "bin1_dist_cm", "tpp_hund", "coord_xfrm")


def stream_name(cfg: dict, has_vb: bool) -> str:
    return (f"c{cfg['n_cells']}x{cfg['n_beams']}_cs{cfg['cell_size_cm']}"
            f"_p{cfg['pings_per_ens']}" + ("_vb" if has_vb else ""))


def iter_decode(idx: FileIndex, chunk_bytes: float = 150e6, leaders_only: bool = False):
    """Genera (nome_stream, cfg, dict_di_array) a blocchi, per ogni layout del file.

    Gli ensemble con checksum errata vengono scartati (conteggio in idx.stats).
    """
    buf = np.frombuffer(idx._mm, np.uint8)
    idx.stats.setdefault("bad_checksum", 0)
    idx.stats.setdefault("bad_layout", 0)
    for li, lay in enumerate(idx.layouts):
        pos = idx.positions[idx.layout_id == li]
        if not lay.valid or lay.find("fixed_leader") is None or lay.find("variable_leader") is None:
            idx.stats["bad_layout"] += len(pos)
            continue
        L = lay.nbytes + 2
        n_chunk = max(1000, int(chunk_bytes // L))
        for i0 in range(0, len(pos), n_chunk):
            p = pos[i0:i0 + n_chunk]
            M = np.stack([buf[q:q + L] for q in p])
            cs = M[:, :lay.nbytes].sum(axis=1, dtype=np.uint64) & 0xFFFF
            ok = cs == _v(M, lay.nbytes, "<u2")
            idx.stats["bad_checksum"] += int((~ok).sum())
            if not ok.all():
                M = M[ok]
                p = p[ok]
            if len(M) == 0:
                continue
            if leaders_only:
                d = {}
                d.update(_decode_fixed(M, *lay.find("fixed_leader")))
                d.update(_decode_variable(M, *lay.find("variable_leader")))
            else:
                d = decode_matrix(M, lay)
            d["file_offset"] = p
            # separa per configurazione (es. stesso layout ma ping/ensemble diversi)
            keys = np.stack([np.asarray(d[k], np.int64) for k in CONFIG_KEYS], axis=1)
            ukeys, inv = np.unique(keys, axis=0, return_inverse=True)
            has_vb = lay.find("vb_echo") is not None or lay.find("vb_velocity") is not None
            for j, uk in enumerate(ukeys):
                sel = inv.ravel() == j
                cfg = {k: int(v) for k, v in zip(CONFIG_KEYS, uk)}
                sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(sel) else v)
                       for k, v in d.items()}
                yield stream_name(cfg, has_vb), cfg, sub


# --------------------------------------------------------------------------
# conversione a xarray
# --------------------------------------------------------------------------
TS_VARS = dict(
    ens_num=("", "numero ensemble"), heading=("deg", "heading"), pitch=("deg", "pitch"),
    roll=("deg", "roll"), temperature=("degC", "temperatura acqua"),
    pressure=("dbar", "pressione"), pressure_var=("dbar", "varianza pressione (raw)"),
    depth_xducer=("m", "profondita' trasduttore (variable leader)"),
    sound_speed=("m s-1", "velocita' del suono"), salinity=("ppt", "salinita'"),
    heading_std=("deg", "dev. std heading"), pitch_std=("deg", "dev. std pitch"),
    roll_std=("deg", "dev. std roll"), bit=("", "built-in test"),
    error_status=("", "error status word"), file_offset=("byte", "posizione nel file"),
)

CONST_ATTRS = ("fw_ver", "fw_rev", "sysconfig", "n_beams", "n_cells", "pings_per_ens",
               "cell_size_cm", "blank_cm", "prof_mode", "corr_threshold", "pg_min",
               "err_vel_max_mm", "tpp_hund", "coord_xfrm", "heading_align", "heading_bias",
               "sensor_source", "sensors_avail", "bin1_dist_cm", "xmit_pulse_cm", "serial",
               "beam_angle")


def describe_config(attrs: dict) -> dict:
    """Interpreta i campi del fixed leader."""
    sc = int(attrs.get("sysconfig", 0))
    ex = int(attrs.get("coord_xfrm", 0))
    ba = attrs.get("beam_angle")
    if ba in (None, 0):
        ba = {0: 15, 1: 20, 2: 30}.get((sc >> 8) & 3, 25)
    return dict(
        orientation="up" if (sc >> 7) & 1 else "down",
        convex=int((sc >> 3) & 1),
        coord_system=COORD_NAMES[(ex >> 3) & 3],
        tilts_used=int((ex >> 2) & 1), three_beam=int((ex >> 1) & 1), bin_mapping=int(ex & 1),
        beam_angle_deg=int(ba),
    )


def to_dataset(d: dict, source: str = ""):
    import xarray as xr

    t = d["time"]
    n = len(t)
    a0 = {k: int(np.asarray(d[k])[0]) for k in CONST_ATTRS if k in d}
    attrs = dict(a0)
    attrs.update(describe_config(a0))
    attrs["source_file"] = os.path.basename(source)
    dv = {}
    for k, (u, ln) in TS_VARS.items():
        if k in d:
            dv[k] = ("time", np.asarray(d[k]), dict(units=u, long_name=ln))
    coords = dict(time=("time", t))
    if "vel" in d:
        nc, nb = d["vel"].shape[1:]
        rng = (a0["bin1_dist_cm"] + np.arange(nc) * a0["cell_size_cm"]) / 100.0
        coords["cell"] = ("cell", np.arange(1, nc + 1))
        coords["range"] = ("cell", rng.astype(np.float32),
                           dict(units="m", long_name="distanza verticale dal trasduttore (centro cella)"))
        coords["beam"] = ("beam", np.arange(1, nb + 1))
        dv["vel"] = (("time", "cell", "beam"), d["vel"],
                     dict(units="m s-1", long_name=f"velocita' ({attrs['coord_system']})"))
        for k, ln in (("corr", "correlazione"), ("echo", "intensita' eco"), ("pg", "percent good")):
            if k in d:
                dv[k] = (("time", "cell", "beam"), d[k], dict(units="counts" if k != "pg" else "%",
                                                              long_name=ln))
    vbk = [k for k in ("vb_vel", "vb_corr", "vb_echo", "vb_pg") if k in d]
    if vbk:
        nvb = d[vbk[0]].shape[1]
        cs = int(d["vb_cell_size_cm"][0]) if "vb_cell_size_cm" in d else a0.get("cell_size_cm", 100)
        r0 = int(d["vb_first_cell_cm"][0]) if "vb_first_cell_cm" in d else a0.get("bin1_dist_cm", 0)
        coords["vb_cell"] = ("vb_cell", np.arange(1, nvb + 1))
        coords["vb_range"] = ("vb_cell", ((r0 + np.arange(nvb) * cs) / 100.0).astype(np.float32),
                              dict(units="m", long_name="distanza dal trasduttore, beam verticale"))
        attrs["vb_cell_size_cm"] = cs
        attrs["vb_first_cell_cm"] = r0
        for k in vbk:
            dv[k] = (("time", "vb_cell"), d[k])
    ds = xr.Dataset(dv, coords=coords, attrs=attrs)
    return ds


NC_ENCODING = {
    "vel": dict(dtype="int16", scale_factor=0.001, _FillValue=-32768),
    "vb_vel": dict(dtype="int16", scale_factor=0.001, _FillValue=-32768),
}


def encoding_for(ds) -> dict:
    enc = {}
    for k, v in ds.data_vars.items():
        e = dict(zlib=True, complevel=1)
        e.update(NC_ENCODING.get(k, {}))
        if v.dtype.kind == "f" and k not in NC_ENCODING:
            e["_FillValue"] = np.nan
        enc[k] = e
    enc["time"] = dict(units="microseconds since 2000-01-01", dtype="int64")
    return enc
