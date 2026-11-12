"""Stile comune dei grafici (matplotlib, PNG) e panoramica dei sensori ausiliari."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# palette: categoriali (ordine fisso), rampa sequenziale blu, divergente blu-grigio-rosso
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#8f2424"])
SEQ = SEQ.with_extremes(bad="#f0efec")
DIV = DIV.with_extremes(bad="#f0efec")

warnings.filterwarnings("ignore", message="(Mean of empty slice|Degrees of freedom|All-NaN slice)")

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "axes.formatter.useoffset": False,
})


def _bin_series(t, y, nbins=3000):
    """Media su intervalli regolari di tempo (lascia NaN nei buchi -> linee interrotte)."""
    t = np.asarray(t)
    if len(t) <= nbins:
        return t, y
    tf = t.astype("datetime64[s]").astype(np.int64).astype(float)
    edges = np.linspace(tf.min(), tf.max() + 1, nbins + 1)
    i = np.clip(np.searchsorted(edges, tf, side="right") - 1, 0, nbins - 1)
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    ok = np.isfinite(y)
    s = np.zeros((nbins,) + y.shape[1:])
    c = np.zeros_like(s)
    np.add.at(s, i, np.where(ok, y, 0))
    np.add.at(c, i, ok)
    with np.errstate(invalid="ignore"):
        m = s / c
    tc = ((edges[:-1] + edges[1:]) / 2).astype("datetime64[s]")
    return tc, m.squeeze()


MESI = ("gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic")
_MONTHS_EN = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _it(s: str) -> str:
    """Abbreviazioni dei mesi in italiano (strftime le scrive nella lingua di sistema, di norma inglese)."""
    for en, it in zip(_MONTHS_EN, MESI):
        s = s.replace(en, it)
    return s


def data_it(t, year: bool = False, hm: bool = False) -> str:
    """Data in italiano: '9 ott', '9 ott 2026', '9 ott 22:00'."""
    t = pd.Timestamp(t)
    s = f"{t.day} {MESI[t.month - 1]}"
    if year:
        s += f" {t.year}"
    if hm:
        s += f" {t:%H:%M}"
    return s


class _ItalianDates(mdates.ConciseDateFormatter):
    def format_ticks(self, values):
        return [_it(s) for s in super().format_ticks(values)]

    def get_offset(self):
        return _it(super().get_offset())


def _fmt_time(ax, maxticks=None):
    """Asse dei tempi con date concise in italiano; ``maxticks`` limita le etichette nei pannelli stretti."""
    loc = mdates.AutoDateLocator(minticks=3, maxticks=maxticks) if maxticks else mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(_ItalianDates(
        loc, zero_formats=["", "%Y", "%b", "%d %b", "%H:%M", "%H:%M"],
        offset_formats=["", "%Y", "%b %Y", "%d %b %Y", "%d %b %Y", "%d %b %Y %H:%M"]))


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  grafico: {path}")


def plot_overview(ds, path, title=""):
    """Sensori ausiliari: pressione, temperatura, heading, pitch/roll."""
    panels = [("pressure", "Pressione [dbar]"), ("temperature", "Temperatura [°C]"),
              ("heading", "Heading [°]")]
    panels = [p for p in panels if p[0] in ds]
    fig, axs = plt.subplots(len(panels) + 1, 1, figsize=(10, 1.9 * (len(panels) + 1)), sharex=True)
    t = ds.time.values
    for ax, (k, lab) in zip(axs, panels):
        tb, yb = _bin_series(t, ds[k].values)
        ax.plot(tb, yb, color=C1)
        ax.set_ylabel(lab)
        ax.ticklabel_format(axis="y", useOffset=False)
    ax = axs[-1]
    for k, c in (("pitch", C1), ("roll", C2)):
        tb, yb = _bin_series(t, ds[k].values)
        ax.plot(tb, yb, color=c, label=k)
        ax.annotate(k, (tb[-1], yb[np.isfinite(yb)][-1] if np.isfinite(yb).any() else 0),
                    xytext=(4, 0), textcoords="offset points", color=INK2, va="center")
    ax.set_ylabel("Inclinazione [°]")
    ax.legend(loc="best", ncols=2)
    _fmt_time(ax)
    axs[0].set_title(title or "Sensori ausiliari")
    _save(fig, path)
