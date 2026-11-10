"""Trasformazioni di coordinate (convenzioni Teledyne RDI, "ADCP Coordinate Transformation").

ATTENZIONE: verificare sul proprio strumento. Se il Sentinel V ha registrato già in
coordinate Earth (EX11111, caso più comune) queste funzioni non vengono usate.
"""
import warnings

import numpy as np


def theoretical_matrix(beam_angle_deg: float, convex: bool = True) -> np.ndarray:
    """Matrice beam -> strumento teorica RDI (righe: X, Y, Z, errore)."""
    th = np.deg2rad(beam_angle_deg)
    a, b = 1.0 / (2.0 * np.sin(th)), 1.0 / (4.0 * np.cos(th))
    d = a / np.sqrt(2.0)
    c = 1.0 if convex else -1.0
    return np.array([[c * a, -c * a, 0, 0], [0, 0, -c * a, c * a], [b, b, b, b], [d, d, -d, -d]])


def instrument_matrix(attrs: dict):
    """Matrice registrata dallo strumento (data type 0x3200), se presente e plausibile."""
    T = attrs.get("transformation_matrix")
    if T is None:
        return None
    T = np.asarray(T, float).reshape(4, 4)
    mx = np.abs(T).max()
    if not np.isfinite(T).all() or not (0.3 < mx < 3.0):
        warnings.warn(f"matrice di trasformazione del file non plausibile (max {mx:.3g}): uso quella teorica")
        return None
    return T


def beam_to_inst(vb: np.ndarray, beam_angle_deg: float, convex: bool = True, tmat=None) -> np.ndarray:
    """vb[..., 4] (beam 1-4) -> [..., 4] (X, Y, Z, errore). Solo soluzioni a 4 beam.

    Se ``tmat`` (4x4) è dato si usa quella (es. la matrice registrata dallo strumento)."""
    if tmat is not None:
        return np.einsum("ij,...j->...i", np.asarray(tmat, np.float32),
                         vb[..., :4].astype(np.float32)).astype(np.float32)
    th = np.deg2rad(beam_angle_deg)
    a = 1.0 / (2.0 * np.sin(th))
    b = 1.0 / (4.0 * np.cos(th))
    d = a / np.sqrt(2.0)
    c = 1.0 if convex else -1.0
    b1, b2, b3, b4 = (vb[..., i] for i in range(4))
    out = np.stack([c * a * (b1 - b2), c * a * (b4 - b3), b * (b1 + b2 + b3 + b4),
                    d * (b1 + b2 - b3 - b4)], axis=-1)
    return out.astype(np.float32)


def inst_to_earth_matrix(heading, pitch, roll, orientation="up"):
    """Matrici di rotazione (N, 3, 3) strumento -> (Est, Nord, Su)."""
    h = np.deg2rad(np.asarray(heading, float))
    r = np.deg2rad(np.asarray(roll, float))
    p = np.arctan(np.tan(np.deg2rad(np.asarray(pitch, float))) * np.cos(r))
    if orientation == "up":
        r = r + np.pi
    ch, sh, cp, sp, cr, sr = np.cos(h), np.sin(h), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    M = np.empty(h.shape + (3, 3))
    M[..., 0, 0] = ch * cr + sh * sp * sr
    M[..., 0, 1] = sh * cp
    M[..., 0, 2] = ch * sr - sh * sp * cr
    M[..., 1, 0] = -sh * cr + ch * sp * sr
    M[..., 1, 1] = ch * cp
    M[..., 1, 2] = -sh * sr - ch * sp * cr
    M[..., 2, 0] = -cp * sr
    M[..., 2, 1] = sp
    M[..., 2, 2] = cp * cr
    return M


def inst_to_earth(vi, heading, pitch, roll, orientation="up"):
    """vi[time, cell, 4] (X, Y, Z, err) -> (E, N, U, err)."""
    M = inst_to_earth_matrix(heading, pitch, roll, orientation)
    enu = np.einsum("tij,tcj->tci", M, vi[..., :3])
    return np.concatenate([enu, vi[..., 3:4]], axis=-1).astype(np.float32)


def to_earth(vel, attrs, heading, pitch, roll, orientation=None):
    """Porta ``vel`` [time, cell, beam] in coordinate Earth, qualunque sia il sistema registrato."""
    cs = attrs.get("coord_system", "earth")
    orientation = orientation or attrs.get("orientation", "up")
    if cs == "earth":
        return vel
    if cs == "ship":
        raise NotImplementedError("coordinate 'ship': rotazione non implementata")
    if cs == "beam":
        vel = beam_to_inst(vel[..., :4], attrs.get("beam_angle_deg", 25), bool(attrs.get("convex", 1)),
                           tmat=instrument_matrix(attrs))
    return inst_to_earth(vel, heading, pitch, roll, orientation)


def rotate_declination(u, v, decl_deg):
    """Ruota (u, v) di ``decl_deg`` (positiva verso Est) da Nord magnetico a Nord geografico."""
    if not decl_deg:
        return u, v
    a = np.deg2rad(decl_deg)
    return u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)
