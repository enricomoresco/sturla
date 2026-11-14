"""Esegue tutta l'analisi del rimescolamento, nell'ordine giusto.

  python analisi/rimescolamento/esegui_tutto.py --dati dati_esterni --risultati out --uscita out_rimescolamento

--dati       cartella con flussi ERA5, vento CMEMS e i file MedFS (temperatura, salinita', mixed layer;
             facoltativi i file del riquadro scaricati con scarica_dati.sh)
--risultati  cartella dei risultati dell'ADCP (quella con preprocessing/ e postprocessing/)
--uscita     dove scrivere tabelle e figure

Ci vogliono alcuni minuti, soprattutto per gli esperimenti con il modello 1D.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import time

import comune as C  # noqa: F401  (imposta i percorsi di import)
import acqua_bassa
import circolazione
import contributi
import esperimenti_1d
import figure
import forzanti
import profili_modello
import richardson
import scomposizione_giornaliera
import trasporto_calore

PASSI = [("forzanti orari", forzanti), ("profili del modello", profili_modello),
         ("esperimenti con il modello 1D", esperimenti_1d), ("scomposizione locale/trasporto", scomposizione_giornaliera),
         ("tabella dei contributi", contributi), ("circolazione e downwelling", circolazione),
         ("numero di Richardson", richardson), ("effetti di acqua bassa", acqua_bassa),
         ("trasporto di calore nel modello (facoltativo)", trasporto_calore), ("figure", figure)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for nome, mod in PASSI:
        t = time.time()
        print(f"\n=== {nome}")
        try:
            mod.main(argv)
        except FileNotFoundError as e:
            if mod is not trasporto_calore:
                raise
            print(f"    saltato: {e} (servono i file del riquadro MedFS, vedi scarica_dati.sh)")
        print(f"    ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
