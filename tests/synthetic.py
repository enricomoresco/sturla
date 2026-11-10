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
