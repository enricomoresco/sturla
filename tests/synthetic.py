"""Generatore di file PD0 sintetici (correnti + burst onde con beam verticale) per i test."""
import struct
from datetime import datetime, timedelta

import numpy as np

RHO, G = 1025.0, 9.81


def _fixed(ncells, nbeams, pings, cs_cm, bin1_cm, ex=0b00011111, up=True):
    b = bytearray(59)
    struct.pack_into("<H", b, 0, 0x0000)
    b[2], b[3] = 47, 19
    sc = 0x4000 | (0x80 if up else 0) | 0x08 | 0x02  # 4-beam janus, convex, up
    struct.pack_into("<H", b, 4, sc)
    b[8], b[9] = nbeams, ncells
    struct.pack_into("<HHH", b, 10, pings, cs_cm, 50)
    b[16], b[17], b[19] = 1, 64, 0
    struct.pack_into("<H", b, 20, 2000)
    b[25] = ex
    struct.pack_into("<HH", b, 32, bin1_cm, 50)
    struct.pack_into("<I", b, 54, 12345)
    b[58] = 25
    return bytes(b)


def _variable(n, t, p_dbar, head=45.0, pitch=1.0, roll=-0.5, temp=15.0):
    b = bytearray(65)
    struct.pack_into("<H", b, 0, 0x0080)
    struct.pack_into("<H", b, 2, n & 0xFFFF)
    b[4:11] = bytes([t.year % 100, t.month, t.day, t.hour, t.minute, t.second, t.microsecond // 10000])
    b[11] = (n >> 16) & 0xFF
    struct.pack_into("<HH", b, 14, 1500, int(p_dbar * 10))
    struct.pack_into("<Hhh", b, 18, int(head * 100), int(pitch * 100), int(roll * 100))
    struct.pack_into("<Hh", b, 24, 38, int(temp * 100))
    struct.pack_into("<II", b, 48, int(round(p_dbar * 1000)), 0)
    b[57:65] = bytes([t.year // 100, t.year % 100, t.month, t.day, t.hour, t.minute, t.second,
                      t.microsecond // 10000])
    return bytes(b)


def make_ensemble(blocks):
    ndt = len(blocks)
    hl = 6 + 2 * ndt
    offs, o = [], hl
    for blk in blocks:
        offs.append(o)
        o += len(blk)
    nbytes = o
    head = struct.pack("<BBHBB", 0x7F, 0x7F, nbytes, 0, ndt) + struct.pack(f"<{ndt}H", *offs)
    body = head + b"".join(blocks)
    return body + struct.pack("<H", sum(body) & 0xFFFF)


def _arr_block(id_, arr, fmt):
    return struct.pack("<H", id_) + np.asarray(arr).astype(fmt).tobytes()


def profile_blocks(ncells, nbeams, vel_mm, corr, echo, pg):
    return [_arr_block(0x0100, vel_mm.ravel(), "<i2"), _arr_block(0x0200, corr.ravel(), "u1"),
            _arr_block(0x0300, echo.ravel(), "u1"), _arr_block(0x0400, pg.ravel(), "u1")]


def vb_blocks(nvb, cs_cm, r0_cm, vel_mm, corr, echo, pg):
    lead = bytearray(42)
    struct.pack_into("<HHHHH", lead, 0, 0x0F01, nvb, 1, cs_cm, r0_cm)
    return [bytes(lead), _arr_block(0x0A00, vel_mm, "<i2"), _arr_block(0x0B00, corr, "u1"),
            _arr_block(0x0C00, echo, "u1"), _arr_block(0x0D00, pg, "u1")]


def make_file(path, hours=12, depth=15.0, hab=0.5, Hs=1.0, Tp=8.0, dir_from=225.0,
              fs=2.0, nburst=1024, seed=0, garbage=True):
    """Scrive un PD0 sintetico. Restituisce i parametri veri."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2025, 3, 1, 0, 0, 0)
    xd = depth - hab                                  # profondità trasduttore
    # correnti: 1 ensemble ogni 10 min, 20 celle da 1 m
    nc, nb, cs, b1 = 20, 4, 100, 150
    rng_c = (b1 + np.arange(nc) * cs) / 100
    # onde: 2 Hz, 8 celle 4 beam + 30 celle beam verticale da 0.5 m
    ncw, csw, b1w = 8, 100, 150
    nvb, csvb, r0vb = 40, 50, 100
    rng_vb = (r0vb + np.arange(nvb) * csvb) / 100
    k = (2 * np.pi / Tp) ** 2 / G
    for _ in range(50):
        k = (2 * np.pi / Tp) ** 2 / (G * np.tanh(k * depth))
    w = 2 * np.pi / Tp
    a_amp = Hs / (2 * np.sqrt(2))                     # sinusoide con Hm0 = Hs
    th_to = np.deg2rad(270 - dir_from)                # direzione verso cui va (matematica)
    events = []
    n_ens = 0
    for h in range(hours):
        tb = t0 + timedelta(hours=h)
        # burst onde
        tt = np.arange(nburst) / fs
        phase = rng.uniform(0, 2 * np.pi)
        eta = a_amp * np.cos(w * tt + phase) + 0.01 * rng.standard_normal(nburst)
        Kp = np.cosh(k * hab) / np.cosh(k * depth)
        p = (RHO * G * (xd + Kp * eta)) / 1e4
        for i in range(nburst):
            z = rng_c[:ncw] + hab
            uorb = a_amp * w * np.cosh(k * z) / np.sinh(k * depth) * np.cos(w * tt[i] + phase)
            vel = np.zeros((ncw, nb))
            vel[:, 0] = uorb * np.cos(th_to) + 0.2
            vel[:, 1] = uorb * np.sin(th_to)
            vel_mm = np.round(vel * 1000).astype(int)
            surf = xd + eta[i]
            echo_vb = (60 + 120 * np.exp(-0.5 * ((rng_vb - surf) / 0.4) ** 2)).astype(int)
            blocks = [_fixed(ncw, nb, 1, csw, b1w), _variable(n_ens, tb + timedelta(seconds=tt[i]), p[i])]
            blocks += profile_blocks(ncw, nb, vel_mm, np.full((ncw, nb), 120), np.full((ncw, nb), 90),
                                     np.tile([0, 0, 0, 100], (ncw, 1)))
            blocks += vb_blocks(nvb, csvb, r0vb, np.zeros(nvb), np.full(nvb, 120), echo_vb, np.full(nvb, 100))
            events.append((tb + timedelta(seconds=tt[i]), make_ensemble(blocks)))
            n_ens += 1
        # correnti
        for m in range(6):
            tc = tb + timedelta(minutes=10 * m)
            tide = 0.5 * np.sin(2 * np.pi * (h + m / 6) / 12.42)
            vel = np.zeros((nc, nb))
            prof = (rng_c / depth) ** (1 / 7)
            vel[:, 0] = tide * prof
            vel[:, 1] = 0.1 * prof
            vel[:, 3] = 0.01 * rng.standard_normal(nc)
            vel_mm = np.round(vel * 1000).astype(int)
            above = rng_c > xd * np.cos(np.deg2rad(25))
            vel_mm[above] = -32768
            corr = np.full((nc, nb), 110)
            corr[above] = 20
            blocks = [_fixed(nc, nb, 60, cs, b1), _variable(n_ens, tc, RHO * G * xd / 1e4)]
            blocks += profile_blocks(nc, nb, vel_mm, corr, np.full((nc, nb), 80),
                                     np.tile([0, 0, 0, 100], (nc, 1)))
            events.append((tc + timedelta(microseconds=1), make_ensemble(blocks)))
            n_ens += 1
    events.sort(key=lambda e: e[0])
    with open(path, "wb") as f:
        for i, (_, e) in enumerate(events):
            if garbage and i == 100:
                f.write(b"\x00\x7f\x7f\x13garbage")          # byte spuri: risincronizzazione
            if garbage and i == 200:
                e = bytearray(e); e[40] ^= 0xFF; e = bytes(e)  # checksum errata
            f.write(e)
        if garbage:
            f.write(events[0][1][:50])                          # ensemble troncato in coda
    return dict(Hs=Hs, Tp=Tp, dir_from=dir_from, depth=depth, hab=hab, n_events=len(events),
                hours=hours)


def make_sentinel_like(path, bursts=8, interval_min=30, nping=1024, fs=2.0, depth=20.0, hab=0.5,
                       Hs=1.0, Tp=6.0, dir_from=200.0, U=0.2, V=-0.1,
                       heading=30.0, pitch=2.0, roll=-3.0, seed=0):
    """Come il file reale ispezionato: un solo stream di ping singoli a 2 Hz in burst,
    coordinate BEAM, beam verticale, matrice di trasformazione (0x3200), blocchi Sentinel
    sconosciuti, niente percent good, fixed/variable leader più lunghi."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from adcp_pd0.transform import theoretical_matrix, inst_to_earth_matrix

    rng_ = np.random.default_rng(seed)
    t0 = datetime(2026, 7, 10)
    xd = depth - hab
    cs, b1, csvb, r0vb = 50, 168, 50, 100
    nc = int((xd + 2 - b1 / 100) / 0.5) + 1          # le celle arrivano oltre la superficie
    nvb = int((xd + 3 - r0vb / 100) / 0.5) + 1
    r_c = (b1 + np.arange(nc) * cs) / 100
    r_vb = (r0vb + np.arange(nvb) * csvb) / 100
    T = theoretical_matrix(25, True)
    Tinv = np.linalg.inv(T)
    tm_block = struct.pack("<H", 0x3200) + np.round(T.ravel() * 10000).astype("<i2").tobytes()
    k = (2 * np.pi / Tp) ** 2 / G
    for _ in range(60):
        k = (2 * np.pi / Tp) ** 2 / (G * np.tanh(k * depth))
    w = 2 * np.pi / Tp
    amp = Hs / (2 * np.sqrt(2))
    th_to = np.deg2rad(270 - dir_from)
    z = r_c + hab                                   # quota sul fondo
    prof = (z / depth) ** (1 / 7)
    side = r_c > xd * np.cos(np.deg2rad(25)) - 0.5
    n = 0
    with open(path, "wb") as f:
        for b in range(bursts):
            tb = t0 + timedelta(minutes=interval_min * b)
            ph = rng_.uniform(0, 2 * np.pi)
            for i in range(nping):
                tt = i / fs
                eta = amp * np.cos(w * tt + ph)
                uo = amp * w * np.cosh(k * z) / np.sinh(k * depth) * np.cos(w * tt + ph)
                wo = amp * w * np.sinh(k * z) / np.sinh(k * depth) * np.sin(w * tt + ph)
                enu = np.stack([U * prof + uo * np.cos(th_to), V * prof + uo * np.sin(th_to), wo], -1)
                h_, p_, r_ = heading + rng_.normal(0, 0.5), pitch, roll
                M = inst_to_earth_matrix([h_], [p_], [r_], "up")[0]
                inst = enu @ M                      # = M^T enu
                inst4 = np.concatenate([inst, np.zeros((nc, 1))], -1)
                beam = inst4 @ Tinv.T
                beam += rng_.normal(0, 0.01, beam.shape)
                vel_mm = np.round(beam * 1000).astype(int)
                vel_mm[side] = -32768
                corr = np.full((nc, 4), 110)
                corr[side] = 30
                p_dbar = RHO * G * (xd + eta * np.cosh(k * hab) / np.cosh(k * depth)) / 1e4
                fl = _fixed(nc, 4, 1, cs, b1, ex=0) + b"\x00"         # 60 B
                vl = _variable(n, tb + timedelta(seconds=tt), p_dbar, h_, p_, r_) + b"\x00"  # 66 B
                echo_vb = (60 + 120 * np.exp(-0.5 * ((r_vb - (xd + eta)) / 0.4) ** 2)).astype(int)
                vbv = np.interp(r_vb + hab, z, wo)
                lead = bytearray(40)
                struct.pack_into("<HHHHH", lead, 0, 0x0F01, nvb, 1, csvb, r0vb)
                blocks = [fl, vl, _arr_block(0x0100, vel_mm.ravel(), "<i2"),
                          _arr_block(0x0200, corr.ravel(), "u1"),
                          _arr_block(0x0300, np.full(nc * 4, 80), "u1"), bytes(lead),
                          _arr_block(0x0A00, np.round(vbv * 1000), "<i2"),
                          _arr_block(0x0B00, np.full(nvb, 110), "u1"),
                          _arr_block(0x0C00, echo_vb, "u1"),
                          struct.pack("<H", 0x7000) + bytes(14), tm_block,
                          struct.pack("<H", 0x7004) + bytes(8)]
                f.write(make_ensemble(blocks))
                n += 1
    return dict(Hs=Hs, Tp=Tp, dir_from=dir_from, U=U, V=V, depth=depth, hab=hab, bursts=bursts,
                prof=prof, side=side, n=n)


def make_trashy(path, **kw):
    """File reale-simile con spazzatura in testa e in coda: byte casuali + ensemble validi
    ma registrati fuori acqua (pressione ~0, tempi prima/dopo il deployment)."""
    import os
    import tempfile
    rng_ = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.pd0")
        truth = make_sentinel_like(good, **kw)
        dry = os.path.join(d, "dry.pd0")
        # stesso strumento ma in aria: pressione ~0, 2 burst
        make_sentinel_like(dry, bursts=2, depth=0.6, hab=0.5, Hs=0.0, seed=5)
        body, dry_b = open(good, "rb").read(), open(dry, "rb").read()
    trash = lambda n: rng_.integers(0, 256, n, dtype=np.uint8).tobytes()
    head = trash(300_000) + dry_b + trash(50_000)
    tail = trash(70_000) + dry_b + trash(300_000)
    with open(path, "wb") as f:
        f.write(head + body + tail)
    truth.update(head_bytes=len(head), tail_bytes=len(tail), body_bytes=len(body))
    return truth
