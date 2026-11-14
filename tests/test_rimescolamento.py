"""Test del modello 1D a bilancio di energia usato nell'analisi del rimescolamento."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analisi" / "rimescolamento"))
from col1d import Column, rho_eos, g, rho0, cp  # noqa: E402

H = 34.5
SPENTO = dict(vento=False, onde=False, fondo=False, convezione=False, Kbg=0.0)


def due_strati(h1=24.0, T1=24.0, T2=20.0, S=38.0):
    """Colonna con strato superiore omogeneo di spessore h1 e strato inferiore piu' freddo."""
    z = np.array([0.0, h1 - 0.01, h1 + 0.01, H])
    return Column(H, z, np.array([T1, T1, T2, T2]), np.full(4, S))


def forzante(**kw):
    f = dict(sw=0.0, lw=0.0, lat=0.0, sen=0.0, EmP=0.0, U10=0.0, ustar=0.0, Us0=0.0, Hm0=0.0, Ub=0.0, Umean=0.0)
    f.update(kw)
    return pd.Series(f)


def test_convezione_rende_stabile_la_colonna():
    z = np.array([0.0, 5.0, 5.01, H])
    col = Column(H, z, np.array([18.0, 18.0, 24.0, 24.0]), np.full(4, 38.0))   # acqua fredda sopra
    liberata = col.convect()
    assert liberata > 0
    assert np.all(np.diff(col.rho()) >= -1e-6)


def test_energia_pari_alla_stratificazione_mescola_tutto():
    col = due_strati()
    phi0 = col.phiH()
    dr = rho_eos(20.0, 38.0) - rho_eos(24.0, 38.0)
    assert abs(phi0 - g * dr * 24.0 * (H - 24.0) / 2) / phi0 < 0.02      # due strati: g*drho*h1*h2/2
    # il costo del mescolamento usa l'equazione di stato linearizzata: differisce di 2-3% da phi*H
    lavoro = col.entrain(1.05 * phi0, from_top=True)
    assert col.phiH() < 0.01 * phi0
    assert lavoro <= 1.05 * phi0


def test_meta_energia_toglie_meta_stratificazione():
    col = due_strati()
    phi0 = col.phiH()
    col.entrain(0.5 * phi0, from_top=True)
    assert abs((phi0 - col.phiH()) - 0.5 * phi0) < 0.03 * phi0
    assert col.phiH() > 0.3 * phi0


def test_raffreddamento_agisce_solo_sotto_lo_strato_mescolato():
    """Raffreddando uno strato mescolato di spessore h la stratificazione cala di rho*B*(H-h)/2 per unita' di tempo."""
    h1, Q, ore = 24.0, -100.0, 24
    col = due_strati(h1=h1)
    phi0 = col.phiH()
    for _ in range(ore):
        col.step(forzante(lw=Q), 3600.0, SPENTO)
    a = -(rho_eos(24.05, 38.0) - rho_eos(23.95, 38.0)) / 0.1 / rho_eos(24.0, 38.0)
    B = g * a * Q / (rho0 * cp)
    atteso = rho0 * B * (H - h1) / 2 * ore * 3600.0
    assert abs((col.phiH() - phi0) - atteso) < 0.05 * abs(atteso)


def test_senza_forzanti_la_colonna_non_cambia():
    col = due_strati()
    T0, S0 = col.T.copy(), col.S.copy()
    for _ in range(5):
        col.step(forzante(), 3600.0, SPENTO)
    assert np.allclose(col.T, T0) and np.allclose(col.S, S0)
