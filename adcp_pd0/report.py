"""Rapporto del pre-processing (Markdown), generato a ogni esecuzione.

Descrive, con i valori effettivi del dataset, tutto ciò che è stato fatto ai dati grezzi per
ottenere i file puliti L2: file di partenza, configurazione dello strumento, periodo in
posizione, movimenti del telaio, trasformazioni, medie, controlli di qualità con le soglie,
statistiche dei flag per causa, file prodotti e limiti noti.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .archive import (VEL_CAUSES, WAVE_CAUSES, TEMP_CAUSES, POS_CAUSES, CAUSE_DESCRIPTIONS, flag_stats,
                      code_version)


def _fmt(x, nd=2, unit=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n.d."
    return f"{x:.{nd}f}{(' ' + unit) if unit else ''}"


def _pct(n, tot):
    return f"{100.0 * n / tot:.1f}%" if tot else "n.d."


def _flag_table(title, flag, cause, causes, unit_name):
    st = flag_stats(flag, cause, causes)
    n = st["n"]
    L = [f"**{title}** ({n:,} {unit_name})", "",
         "| flag | significato | numero | % |", "|---|---|---|---|"]
    names = {1: "buono", 3: "sospetto", 4: "cattivo", 9: "mancante"}
    for v in (1, 3, 4, 9):
        L.append(f"| {v} | {names[v]} | {st['per_flag'][v]:,} | {_pct(st['per_flag'][v], n)} |")
    L += ["", "| causa (bit) | flag | numero | % |", "|---|---|---|---|"]
    for bit, name, sev in causes:
        k = st["per_causa"][name]
        L.append(f"| `{name}` ({bit}) | {sev} | {k:,} | {_pct(k, n)} |")
    L.append("")
    return L


def write_report(path: str, did: str, md: dict, hab: float, inv: dict | None, dep: dict | None,
                 chk: dict | None, c=None, b=None, h=None, files=None, reader=None,
                 p_atm: float = 0.0) -> str:
    inv = inv or {}
    stream = next((s for s in inv.get("streams", {}).values() if s.get("label") in ("waves", "currents")), {})
    sa = stream.get("attrs", {})
    ca = dict(c.attrs) if c is not None else {}
    ba = dict(b.attrs) if b is not None else {}
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Rapporto del pre-processing - {did}", "",
         f"Generato il {now} da `adcp_pd0` ({code_version()}).", "",
         "Descrive tutte le operazioni eseguite sui dati grezzi dell'ADCP per ottenere i file puliti "
         "(livello L2), con le soglie usate e le statistiche dei controlli di qualita'. Le analisi "
         "(spettri d'onda, termoclino, statistiche, grafici) sono nel post-processing.", ""]

    # 1. dati di partenza
    L += ["## 1. Dati di partenza", ""]
    for f, st in inv.get("files", {}).items():
        L += [f"- File `{os.path.basename(f)}`: {st.get('bytes_scanned', 0) / 2**20:,.1f} MB analizzati, "
              f"{st.get('n_ensembles', 0):,} ensemble (ping) letti; checksum errate {st.get('bad_checksum', 0)}, "
              f"risincronizzazioni {st.get('n_resync', 0)}, byte non riconosciuti {st.get('bytes_skipped', 0):,}."]
    hb = sa.get("heading_bias")
    L += ["", "Strumento e configurazione (dal fixed leader del formato PD0):", "",
          f"- Teledyne RDI Sentinel V, numero di serie {sa.get('serial', 'n.d.')}, firmware "
          f"{sa.get('fw_ver', '?')}.{sa.get('fw_rev', '?')}",
          f"- orientamento: {sa.get('orientation', 'n.d.')}; angolo dei beam {sa.get('beam_angle_deg', 'n.d.')} gradi; "
          f"trasduttore a {hab} m dal fondale (parametro `--hab` o metadati)",
          f"- coordinate registrate: {sa.get('coord_system', 'n.d.')}; ping per ensemble "
          f"{sa.get('pings_per_ens', 'n.d.')}",
          f"- beam inclinati: {sa.get('n_cells', 'n.d.')} celle da {sa.get('cell_size_cm', 'n.d.')} cm, "
          f"blank {sa.get('blank_cm', 'n.d.')} cm, centro della prima cella a {sa.get('bin1_dist_cm', 'n.d.')} cm",
          f"- beam verticale: celle da {sa.get('vb_cell_size_cm', 'n.d.')} cm, prima cella a "
          f"{sa.get('vb_first_cell_cm', 'n.d.')} cm",
          (f"- heading bias impostato nello strumento (comando EB): {int(hb) / 100:.2f} gradi. "
           "Il Teledyne applica questa correzione all'heading registrato: di norma e' la declinazione "
           "magnetica, quindi le direzioni sono gia' riferite al Nord geografico e non va aggiunta altra "
           "declinazione (verificare con chi ha configurato lo strumento)."
           if hb not in (None, "", 0, "0") else
           "- heading bias (comando EB): 0, nessuna correzione nello strumento: la declinazione magnetica "
           "va applicata in elaborazione (`--declination`)"), ""]
    if reader is not None:
        from .convert import burst_summary
        bs = burst_summary(reader.times())
        vals, cnt = np.unique(bs["lengths"], return_counts=True)
        top = np.argsort(cnt)[::-1][:4]
        L += [f"Campionamento: un ping ogni {bs['dt_s']:.2f} s durante i burst; {bs['n_bursts']:,} burst. "
              "Lunghezze piu' frequenti: " + ", ".join(
                  f"{vals[i]} ping ({vals[i] * bs['dt_s'] / 60:.1f} min) x {cnt[i]:,}" for i in top)
              + f". Intervallo mediano fra inizi di burst {bs['interval_median_min']:.0f} min.", ""]

    # 2. periodo in posizione
    L += ["## 2. Periodo in posizione e movimenti del telaio", ""]
    if dep:
        cr = dep.get("criteria", {})
        L += [f"- Dati registrati: {dep['data_start'][:19]} -> {dep['data_end'][:19]}",
              f"- **Strumento in posizione: {dep['start'][:19]} -> {dep['end'][:19]}** "
              f"(metodo: {dep.get('method', 'n.d.')})",
              "- Criteri (mediane al minuto): pressione > "
              f"{cr.get('p_min')} dbar; scarto dalla pressione di riferimento ({dep.get('p_ref_dbar')} dbar) < "
              f"{dep.get('tolerance_dbar')} dbar; inclinazione < {cr.get('tilt_max')} gradi e entro "
              f"{cr.get('tilt_change_max')} gradi dal valore di riferimento ({dep.get('tilt_ref_deg')} gradi); "
              f"condizioni vere per almeno {cr.get('min_stable_min')} minuti; margine di "
              f"{cr.get('margin_min')} minuti solo dove c'e' una fase fuori posizione.",
              "- Tutti i dati fuori da questo periodo sono **mantenuti con flag 4** (causa `fuori_posizione`).", ""]
        ev = dep.get("motion_events", [])
        L += [f"Movimenti del telaio (salti fra minuti consecutivi con dati: heading > "
              f"{cr.get('motion_heading_deg')} gradi o pitch/roll > {cr.get('motion_tilt_deg')} gradi; "
              f"eventi a meno di {cr.get('motion_merge_min')} min uniti): **{len(ev)}**. I dati che si "
              "sovrappongono a questi intervalli hanno flag 3 (causa `movimento_telaio`); le velocita' restano "
              "valide perche' ogni ping e' ruotato con il proprio heading, ma la posizione delle celle e il "
              "riferimento dello strumento cambiano durante l'evento.", ""]
        if ev:
            L += ["| inizio | fine | heading prima -> dopo | pitch prima -> dopo | roll prima -> dopo |",
                  "|---|---|---|---|---|"]
            for e in ev:
                L.append(f"| {e['start'][:16]} | {e['end'][:16]} | {_fmt(e['heading_before'], 1)} -> "
                         f"{_fmt(e['heading_after'], 1)} | {_fmt(e['pitch_before'])} -> {_fmt(e['pitch_after'])} | "
                         f"{_fmt(e['roll_before'])} -> {_fmt(e['roll_after'])} |")
            L.append("")
    else:
        L += ["Periodo in posizione non determinato.", ""]
    if reader is not None:
        try:
            p = reader.load(["pressure"]).pressure.values
            air = p[p < 1.0]
            if air.size > 100:
                L += [f"Pressione letta in aria (strumento fuori acqua): mediana {np.median(air):.2f} dbar "
                      f"(5-95%: {np.percentile(air, 5):.2f}-{np.percentile(air, 95):.2f}). E' l'offset del sensore "
                      f"rispetto alla pressione atmosferica; offset sottratto in elaborazione (`--p-atm`): "
                      f"{p_atm} dbar.", ""]
        except Exception:
            pass

    # 3. correnti
    L += ["## 3. Correnti", ""]
    if c is not None:
        tm = sa.get("transformation_matrix")
        tml, same = [], ""
        if tm is not None:
            vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(tm))]
            if len(vals) == 16:
                tml = ["", "   | riga | beam 1 | beam 2 | beam 3 | beam 4 |", "   |---|---|---|---|---|"]
                for name, row in zip(("X", "Y", "Z", "errore"), np.reshape(vals, (4, 4))):
                    tml.append(f"   | {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
                tml.append("")
                from .transform import theoretical_matrix
                th = theoretical_matrix(float(sa.get("beam_angle_deg", 25)), bool(sa.get("convex", 1)))
                dmax = float(np.abs(np.reshape(vals, (4, 4)) - th).max())
                same = (f"; coincide con quella teorica RDI per beam a {sa.get('beam_angle_deg', 25)} gradi "
                        f"(differenza massima {dmax:.4f})" if dmax < 1e-3 else
                        f"; DIVERSA da quella teorica RDI (differenza massima {dmax:.4f})")
        binm = ca.get("proc_avg_bin_minutes")
        if sa.get("coord_system") == "earth":
            L += ["Passi, nell'ordine:", "",
                  "1. Velocita' gia' registrate dallo strumento in coordinate Est/Nord/Su: nessuna trasformazione."]
        else:
            L += ["Passi, nell'ordine:", "",
                  "1. **Trasformazione beam -> strumento** con la matrice registrata dallo strumento "
                  f"(data type 0x3200{same}); solo soluzioni a 4 beam."]
            L += tml
            L += ["2. **Strumento -> Est/Nord/Su** ping per ping con heading, pitch e roll del ping (convenzione "
                  "Teledyne RDI, pitch corretto per il roll, roll + 180 gradi per strumento rivolto verso l'alto)."]
        L += [f"3. **Declinazione magnetica**: {ca.get('proc_declination', 0)} gradi aggiunti in elaborazione "
              f"(metadati: {md.get('magnetic_declination_deg', 'n.d.')}, gia' applicata nello strumento: "
              f"{md.get('magnetic_declination_applied', 'n.d.')}).",
              f"4. **Scarto dei ping** con correlazione < {ca.get('proc_avg_corr_min', ca.get('proc_qc_corr_min', 'n.d.'))} "
              "counts su almeno un beam.",
              "5. **Media** "
              + (f"su intervalli regolari di {float(binm):g} minuti (ore di orologio), tempo al centro "
                 "dell'intervallo, usando solo i ping in posizione; intervalli senza misure come righe vuote."
                 if binm else "gia' fatta dallo strumento (ensemble) o su ogni burst.")
              + f" Celle con meno del {100 * float(ca.get('proc_avg_min_good_frac', 0.5)):.0f}% di ping validi rimosse "
                "(causa `pochi_ping_validi`), fra 50% e 80% sospette (`ping_validi_ridotti`).",
              f"6. **Controlli sulle medie**: velocita' d'errore |err| > {ca.get('proc_qc_err_max')} m/s "
              f"(`velocita_errore_alta`); inclinazione > {ca.get('proc_qc_tilt_max')} gradi (`inclinazione_eccessiva`); "
              "celle con distanza dal trasduttore > D cos(angolo beam) - 1 cella, con D profondita' del "
              "trasduttore dalla pressione (`contaminazione_superficie`).",
              f"7. Quota delle celle sul fondale = distanza dal trasduttore + {hab} m.", ""]
        tin = c.deployment_qc.values == 1
        u = c.u.values
        L += [f"Risultato: {c.sizes['time']:,} profili ({int(tin.sum()):,} in posizione), {c.sizes['z']} quote "
              f"da {float(c.z.min()):.2f} a {float(c.z.max()):.2f} m sul fondale; celle con velocita' valida "
              f"nel periodo in posizione: {_pct(int(np.isfinite(u[tin]).sum()), u[tin].size)}.", ""]
    if chk:
        L += ["**Controllo della trasformazione** (indipendente dalla trasformazione stessa): sotto un'onda la "
              "velocita' verticale e' in fase con dη/dt, dove η e' la superficie tracciata dall'eco del beam "
              f"verticale. Sui {chk['n_bursts']} burst piu' energetici (Hm0 {chk['hm0_range'][0]:.2f}-"
              f"{chk['hm0_range'][1]:.2f} m), cella a {chk['cell_range_m']:.1f} m dal trasduttore, al picco "
              "dello spettro:", "",
              f"- coerenza² fra w dai 4 beam e dη/dt = {_fmt(chk.get('coh2_w4_deta'))}, fase "
              f"{_fmt(chk.get('phase_w4_deta_deg'), 0)} gradi (in fase = trasformazione e orientamento corretti);",
              f"- coerenza² fra velocita' del beam verticale e dη/dt = {_fmt(chk.get('coh2_wvb_deta'))}, fase "
              f"{_fmt(chk.get('phase_wvb_deta_deg'), 0)} gradi (±180 = convenzione positiva verso il trasduttore);",
              f"- coerenza² fra velocita' orizzontali e η = {_fmt(chk.get('coh2_uv_eta'))};",
              f"- sfasamento temporale stimato dei 4 beam rispetto al beam verticale: "
              f"{_fmt(chk.get('lag_w4_s'), 2, 's')}.", ""]

    # 4. burst d'onda
    L += ["## 4. Serie dei burst d'onda", ""]
    if b is not None:
        nmin = ba.get("proc_burst_min_samples", 256)
        L += ["Passi:", "",
              f"1. Si tengono i burst con almeno {nmin} ping ({ba.get('proc_n_short_bursts_skipped', 'n.d.')} burst "
              "piu' corti esclusi: servono solo per le correnti e la temperatura); i burst piu' lunghi di "
              f"{ba.get('proc_burst_max_samples', 4096)} ping sarebbero divisi in tratti consecutivi.",
              "2. **Superficie dal beam verticale**: per ogni ping, massimo dell'eco nella finestra ±35% della "
              "profondita' attesa, con interpolazione parabolica fra le celle; ping senza un picco netto (bordo "
              "della finestra o prominenza < 10 counts) lasciati mancanti (`surface_range`).",
              "3. **Elevazione pulita** (`eta`): valori oltre 4 MAD dalla mediana del burst rimossi, buchi "
              "interpolati se < 15% dei ping, media del burst tolta; se i buchi sono di piu' l'elevazione del "
              "burst resta mancante.",
              f"4. **Pressione** per ping (offset sottratto: {ba.get('proc_p_atm', 0)} dbar).",
              "5. **Velocita' Est/Nord** per ping ad alcune quote (stessa trasformazione delle correnti), "
              "per la stima della direzione nel post-processing.",
              "6. Nessuno spettro e nessun parametro d'onda in questa fase.", ""]
        if "level_range" in b.coords:
            L += ["Quote delle velocita': " + ", ".join(
                f"{float(r):.2f} m dal trasduttore ({float(r) + hab:.2f} m dal fondale)"
                for r in b["level_range"].values) + ".", ""]
        ok = np.isin(b["burst_qc"].values, (1, 3))
        txt = f"Risultato: {b.sizes['time']:,} burst, {int(ok.sum()):,} utilizzabili (flag 1 o 3)"
        if "surface_found_frac" in b and ok.any():
            sf = b["surface_found_frac"].values[ok]
            txt += f"; superficie trovata nel {100 * np.nanmedian(sf):.0f}% dei ping (mediana)"
            if "surface_range_mean" in b:
                dr = np.nanmedian(b["surface_range_mean"].values[ok] - b["depth_sensor"].values[ok])
                txt += (f"; distanza superficie meno profondita' da pressione = {dr:+.2f} m (atteso piccolo: "
                        "distanza verticale fra sensore di pressione e trasduttore, piu' l'offset di pressione)")
        L += [txt + ".", ""]
    else:
        L += ["Nessuno stream a burst trovato.", ""]

    # 5. temperatura
    L += ["## 5. Temperatura", ""]
    if h is not None:
        g = h.qc.values == 1
        L += ["Temperatura del sensore nella testa dello strumento (acqua al fondo); la salinita' non e' misurata "
              "(valore impostato, usato solo per la velocita' del suono). Media oraria su ore di orologio di tutti "
              "i ping in posizione (tempo al centro dell'ora, estremi in `time_bnds`); minimo, massimo, deviazione "
              "standard, numero di ping e minuti campionati (le misure ci sono solo durante i burst). Flag 3 se "
              "meno di 60 ping (30 s) nell'ora.", ""]
        if g.any():
            tg = h.temperature_mean.values[g]
            L += [f"Risultato: {int(g.sum()):,} ore con flag 1; temperatura media oraria fra {np.nanmin(tg):.2f} e "
                  f"{np.nanmax(tg):.2f} gradi C.", ""]

    # 6. flag
    L += ["## 6. Flag di qualita'", "",
          "Schema IOOS/QARTOD per il flag principale: 1 buono, 3 sospetto, 4 cattivo, 9 mancante "
          "(2 = non valutato non usato). Ogni flag ha accanto una variabile `*_cause` che dice **perche'**: "
          "e' la somma dei bit delle cause presenti; il flag principale e' il piu' grave fra le cause. "
          "Le velocita' con flag 4 sono rimosse (mancanti); quelle con flag 3 restano.", "",
          "| causa | significato |", "|---|---|"]
    for k, v in CAUSE_DESCRIPTIONS.items():
        L.append(f"| `{k}` | {v} |")
    L.append("")
    if c is not None:
        L += _flag_table("Velocita' (tutte le celle, comprese quelle sopra la superficie)", c.vel_qc.values,
                         c.vel_qc_cause.values, VEL_CAUSES, "celle x intervalli")
        L += _flag_table("Serie al fondo (temperatura, pressione, assetto) - `deployment_qc`",
                         c.deployment_qc.values, c.deployment_qc_cause.values, POS_CAUSES, "intervalli")
    if b is not None:
        L += _flag_table("Burst d'onda - `burst_qc`", b.burst_qc.values, b.burst_qc_cause.values,
                         WAVE_CAUSES, "burst")
    if h is not None:
        L += _flag_table("Temperatura oraria - `temperature_qc`", h.qc.values, h.qc_cause.values,
                         TEMP_CAUSES, "ore")

    # 7. file
    L += ["## 7. File prodotti", ""]
    desc = {"_L2_currents.nc": "profili di corrente (tempo x quota), flag e serie al fondo",
            "_L2_temperature.nc": "temperatura al fondo, statistiche orarie",
            "_L2_wave_bursts.nc": "serie pulite dei burst d'onda (elevazione, pressione, velocita')",
            "deployment.json": "periodo in posizione, criteri e movimenti del telaio",
            "inventory.json": "resoconto della lettura del file PD0 e configurazione dello strumento",
            "check_transform.json": "risultati del controllo della trasformazione",
            "metadata.json": "metadati usati", "LEGGIMI_DATI.md": "guida ai dati"}
    for f in files or []:
        d = next((v for k, v in desc.items() if f.endswith(k)), "")
        L.append(f"- `{os.path.basename(f)}`" + (f": {d}" if d else ""))
    L += ["", "Tutti i ping decodificati (livello L1) restano in `raw/`; i prodotti derivati (livello L3) e i "
          "grafici sono in `postprocessing/`.", ""]

    # 8. limiti
    L += ["## 8. Limiti e avvertenze", "",
          f"- Fuso orario: {md.get('time_zone', 'non specificato: da verificare (orologio dello strumento)')}.",
          "- I dati sono controllati automaticamente, non validati manualmente.",
          "- La pressione e' assunta relativa (azzerata in aria); l'offset letto in aria e' riportato sopra.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path
