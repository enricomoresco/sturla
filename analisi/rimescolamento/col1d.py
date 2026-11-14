"""Modello 1D a bilancio di energia (tipo Kraus e Turner, 1967; Simpson e Bowers, 1981).

Contiene solo processi locali: flussi di calore e di acqua dolce in superficie (la radiazione
solare penetra con un profilo a due esponenziali, acqua di tipo Jerlov II), convezione,
mescolamento da vento, turbolenza di Langmuir, frangimento, attrito al fondo della corrente e
delle onde, diffusione di fondo. Nessuna avvezione: la differenza con il modello 3D e con le
misure e' il contributo del trasporto (downwelling, correnti).

Energia meccanica disponibile ogni ora (J/m2), usata per inglobare acqua sotto lo strato
mescolato (o sopra lo strato di fondo):
  convezione penetrativa  0.2 x energia potenziale liberata dalla convezione
  vento                   delta * 6.4e-5 * rho_aria * U10^3                (Simpson e Bowers, 1981)
  Langmuir                0.2 * 0.22 * rho * u*^2 * Us0                     (Belcher et al., 2012)
  frangimento             0.2 * 100 * rho * u*^3 * min(1, 0.3 Hs/h)         (Craig e Banner, 1994; Terray et al., 1996)
  attrito della corrente  0.0037 * 2.5e-3 * rho * U^3, dal fondo            (Simpson e Bowers, 1981)
  attrito delle onde      0.0037 * rho * fw/2 * <|u_b|^3>, dal fondo
"""
import numpy as np, pandas as pd

g, rho0, cp, rhoa = 9.81, 1025.0, 3990.0, 1.22


def rho_eos(t, s):
    rw = 999.842594 + 6.793952e-2*t - 9.095290e-3*t**2 + 1.001685e-4*t**3 - 1.120083e-6*t**4 + 6.536332e-9*t**5
    return (rw + s*(0.824493 - 4.0899e-3*t + 7.6438e-5*t**2 - 8.2467e-7*t**3 + 5.3875e-9*t**4)
            + s**1.5*(-5.72466e-3 + 1.0227e-4*t - 1.6546e-6*t**2) + 4.8314e-4*s**2)


class Column:
    def __init__(self, H, z_prof, T_prof, S_prof, dz=0.25, jerlov=(0.77, 1.5, 14.0), bbl=0.0):
        self.H, self.dz = H, dz
        self.n = int(round(H / dz))
        self.zc = (np.arange(self.n) + 0.5) * dz                      # profondita' dei centri
        self.hb = H - self.zc                                            # altezza dal fondo
        self.T = np.interp(self.zc, z_prof, T_prof).astype(float)
        self.S = np.interp(self.zc, z_prof, S_prof).astype(float)
        if bbl > 0:                                                      # strato limite di fondo gia' mescolato
            k = self.zc > H - bbl
            self.T[k], self.S[k] = self.T[k].mean(), self.S[k].mean()
        R, x1, x2 = jerlov
        ze = np.arange(self.n + 1) * dz
        I = R * np.exp(-ze / x1) + (1 - R) * np.exp(-ze / x2)
        self.sw_frac = -np.diff(I)
        nb = max(1, int(round(3.0 / dz)))                                # quello che arriva al fondo scalda i 3 m sopra il fondale
        self.sw_frac[-nb:] += I[-1] / nb

    # ---------------------------------------------------------------- diagnostica
    def rho(self, T=None, S=None):
        return rho_eos(self.T if T is None else T, self.S if S is None else S)

    def pe(self, T=None, S=None):
        return float(np.sum(self.rho(T, S) * g * self.hb) * self.dz)

    def _coeff(self):
        """Coefficienti di dilatazione termica e di contrazione salina (kg/m3 per grado e per psu) alla media della colonna."""
        t, s = float(self.T.mean()), float(self.S.mean())
        a = -(rho_eos(t + 0.05, s) - rho_eos(t - 0.05, s)) / 0.1
        b = (rho_eos(t, s + 0.05) - rho_eos(t, s - 0.05)) / 0.1
        return a, b

    def pe_lin(self, T, S, coeff):
        """Energia potenziale con l'equazione di stato linearizzata: mescolare conserva la massa, quindi il
        costo di un mescolamento non contiene l'effetto spurio della non linearita' (cabbeling) su celle fisse."""
        a, b = coeff
        return float(np.sum((-a * T + b * S) * g * self.hb) * self.dz)

    def phiH(self):
        r = self.rho()
        return float(np.sum((r.mean() - r) * g * self.hb) * self.dz)   # energia per mescolare tutta la colonna, J/m2

    def at(self, depth):
        return float(np.interp(depth, self.zc, self.T))

    def dsig(self):
        r = self.rho(); return float(r[-1] - r[0])

    def mld(self, dsig=0.1, zref=1.0):
        r = self.rho(); r0 = np.interp(zref, self.zc, r)
        k = np.where(r > r0 + dsig)[0]
        return float(self.zc[k[0]]) if len(k) else self.H

    def _top(self):
        r = self.rho(); k = 0
        while k < self.n - 1 and abs(r[k + 1] - r[0]) < 1e-5:
            k += 1
        return k

    def _bot(self):
        r = self.rho(); k = self.n - 1
        while k > 0 and abs(r[k - 1] - r[-1]) < 1e-5:
            k -= 1
        return k

    # ---------------------------------------------------------------- processi
    def convect(self):
        """Aggiustamento convettivo; restituisce l'energia potenziale liberata (J/m2)."""
        co = self._coeff()
        pe0 = self.pe_lin(self.T, self.S, co); T, S = self.T, self.S
        r = rho_eos(T, S); k = 0
        while k < self.n - 1:
            if r[k] > r[k + 1] + 1e-7:
                a, b = k, k + 1
                while True:
                    Tm, Sm = T[a:b + 1].mean(), S[a:b + 1].mean(); rm = rho_eos(Tm, Sm)
                    grown = False
                    if a > 0 and r[a - 1] > rm + 1e-7:
                        a -= 1; grown = True
                    if b < self.n - 1 and rm > r[b + 1] + 1e-7:
                        b += 1; grown = True
                    if not grown:
                        break
                T[a:b + 1], S[a:b + 1] = Tm, Sm
                r = rho_eos(T, S); k = max(a - 1, 0)
            else:
                k += 1
        return pe0 - self.pe_lin(self.T, self.S, co)

    def entrain(self, E, from_top=True):
        """Usa l'energia E (J/m2) per inglobare acqua nello strato mescolato (dall'alto o dal fondo).
        Restituisce il lavoro fatto contro la galleggiabilita'."""
        work = 0.0
        co = self._coeff()
        while E > 1e-9:
            if from_top:
                m = self._top()
                if m >= self.n - 1:
                    break
                sl = slice(0, m + 2)
            else:
                m = self._bot()
                if m <= 0:
                    break
                sl = slice(m - 1, self.n)
            T2, S2 = self.T.copy(), self.S.copy()
            T2[sl], S2[sl] = self.T[sl].mean(), self.S[sl].mean()
            C = self.pe_lin(T2, S2, co) - self.pe_lin(self.T, self.S, co)
            if C <= 0:
                self.T, self.S = T2, S2; continue
            if C <= E:
                self.T, self.S = T2, S2; E -= C; work += C
            else:
                f = E / C
                self.T = self.T + f * (T2 - self.T); self.S = self.S + f * (S2 - self.S); work += E; E = 0.0
        return work

    def diffuse(self, K, dt, nsub=4):
        for _ in range(nsub):
            for X in (self.T, self.S):
                d = np.zeros_like(X)
                flux = K * np.diff(X) / self.dz
                d[:-1] += flux; d[1:] -= flux
                X += d * (dt / nsub) / self.dz

    def step(self, f, dt, opt):
        """Un passo orario. f: forzanti dell'ora (sw, lw, lat, sen, EmP, U10, ustar, Us0, Hm0, Ub, Umean).
        opt: flussi, convezione, vento, onde, fondo (True/False), fattore_vento (moltiplica U10 e u*),
        delta (efficienza del vento), fw (attrito delle onde), Kbg (diffusione di fondo, m2/s).
        Restituisce i contributi alla variazione di phi*H (J/m2)."""
        out = {}
        p0 = self.phiH()
        # flussi in superficie
        if opt.get("flussi", True):
            self.T += f["sw"] * self.sw_frac * dt / (rho0 * cp * self.dz)
            self.T[0] += (f["lw"] + f["lat"] + f["sen"]) * dt / (rho0 * cp * self.dz)
            self.S[0] += self.S[0] * f["EmP"] / 1000.0 * dt / self.dz
        released = self.convect()
        p1 = self.phiH(); out["galleggiamento"] = p1 - p0
        # energia meccanica disponibile in alto, per sorgente
        fw = opt.get("fattore_vento", 1.0)
        ust = f["ustar"] * fw
        h = (self._top() + 1) * self.dz
        src = {
            "convezione": 0.2 * released if opt.get("convezione", True) else 0.0,
            "vento": opt.get("delta", 0.023) * 6.4e-5 * rhoa * (f["U10"] * fw) ** 3 * dt if opt.get("vento", True) else 0.0,
            "Langmuir": 0.2 * 0.22 * rho0 * ust ** 2 * f["Us0"] * dt if opt.get("onde", True) else 0.0,
            "frangimento": 0.2 * 100.0 * rho0 * ust ** 3 * min(1.0, 0.3 * f["Hm0"] / max(h, 0.5)) * dt if opt.get("onde", True) else 0.0,
        }
        Etop = sum(src.values())
        p1 = self.phiH(); self.entrain(Etop, True); p2 = self.phiH()
        for k, v in src.items():
            out[k] = (p2 - p1) * (v / Etop) if Etop > 0 else 0.0
        # fondo: corrente e onde
        sig = f["Ub"] / 2.0
        srcb = {
            "attrito_corrente": 0.0037 * 2.5e-3 * rho0 * f["Umean"] ** 3 * dt if opt.get("fondo", True) else 0.0,
            "onde_fondo": 0.0037 * rho0 * 0.5 * opt.get("fw", 0.02) * 2 * np.sqrt(2 / np.pi) * sig ** 3 * dt if (opt.get("fondo", True) and opt.get("onde", True)) else 0.0,
        }
        Eb = sum(srcb.values())
        self.entrain(Eb, False); p3 = self.phiH()
        for k, v in srcb.items():
            out[k] = (p3 - p2) * (v / Eb) if Eb > 0 else 0.0
        self.diffuse(opt.get("Kbg", 1e-5), dt, nsub=max(4, int(np.ceil(opt.get("Kbg", 1e-5) * dt / self.dz ** 2 / 0.4))))
        self.convect()
        out["diffusione"] = self.phiH() - p3
        return out


def run(col, F, t0, t1, opt, depth_probe):
    rows = []
    dt = 3600.0
    for t, f in F.loc[t0:t1 - pd.Timedelta("1h")].iterrows():
        c = col.step(f, dt, opt)
        c.update(time=t + pd.Timedelta("1h"), T_probe=col.at(depth_probe), mld=col.mld(), phiH=col.phiH(),
                 T_sup=col.at(1.0), dsig=col.dsig())
        rows.append(c)
    return pd.DataFrame(rows).set_index("time")
