#!/usr/bin/env bash
# Scarica i dati esterni che servono a completare l'analisi del rimescolamento a Sturla:
#   1. modello MedFS (Copernicus Marine), orario, in un riquadro attorno alla stazione:
#      correnti (uo, vo), temperatura, salinita', livello del mare; velocita' verticale (wo, solo giornaliera)
#   2. vento misurato all'aeroporto di Genova (METAR LIMJ), archivio dell'Iowa Environmental Mesonet
#
# Uso:
#   bash analisi/rimescolamento/scarica_dati.sh [cartella] [inizio] [fine]
#   (default: dati_esterni 2026-07-01 2026-10-31)
#
# Serve:
#   - per MedFS: pip install copernicusmarine, un account gratuito Copernicus Marine
#     (marine.copernicus.eu) e, una volta sola, il comando: copernicusmarine login
#   - per il vento misurato: solo curl, nessun account
#   - per ERA5 (vento a 10 m, sforzo, pressione al livello del mare): scarica_era5.py
#
# I file gia' scaricati non vengono riscaricati: per rifarli, cancellarli.
set -euo pipefail

DEST="${1:-dati_esterni}"
INIZIO="${2:-2026-07-01}"
FINE="${3:-2026-10-31}"
mkdir -p "$DEST"

# Riquadro: dalla costa a circa 20 km al largo, circa 15 km a est e a ovest della stazione
LON_MIN=8.85; LON_MAX=9.20; LAT_MIN=44.25; LAT_MAX=44.45
PROF_MAX=200    # m: tutta la colonna sulla piattaforma davanti a Genova

medfs() {   # medfs <dataset> <file di uscita> <opzioni: variabili, profondita'>
  local ds="$1" out="$2"; shift 2
  if [ -s "$DEST/$out" ]; then echo "  gia' presente: $out"; return; fi
  echo "  $out"
  copernicusmarine subset --dataset-id "$ds" "$@" \
    --minimum-longitude "$LON_MIN" --maximum-longitude "$LON_MAX" \
    --minimum-latitude "$LAT_MIN" --maximum-latitude "$LAT_MAX" \
    --start-datetime "${INIZIO}T00:00:00" --end-datetime "${FINE}T23:59:59" \
    --coordinates-selection-method outside \
    --output-directory "$DEST" --output-filename "$out" --log-level WARN
}

if command -v copernicusmarine >/dev/null 2>&1; then
  echo "== 1. Modello MedFS (MEDSEA_ANALYSISFORECAST_PHY_006_013)"
  medfs cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m medfs_riquadro_correnti.nc    --variable uo --variable vo --minimum-depth 0 --maximum-depth "$PROF_MAX"
  medfs cmems_mod_med_phy-tem_anfc_4.2km-3D_PT1H-m medfs_riquadro_temperatura.nc --variable thetao --minimum-depth 0 --maximum-depth "$PROF_MAX"
  medfs cmems_mod_med_phy-sal_anfc_4.2km-3D_PT1H-m medfs_riquadro_salinita.nc    --variable so --minimum-depth 0 --maximum-depth "$PROF_MAX"
  medfs cmems_mod_med_phy-ssh_anfc_4.2km-2D_PT1H-m medfs_riquadro_livello.nc     --variable zos
  # la velocita' verticale esiste solo come media giornaliera
  medfs cmems_mod_med_phy-wcur_anfc_4.2km_P1D-m    medfs_riquadro_velocita_verticale_giornaliera.nc --variable wo --minimum-depth 0 --maximum-depth "$PROF_MAX"
else
  echo "== 1. MedFS saltato: manca il comando copernicusmarine (pip install copernicusmarine, poi copernicusmarine login)"
fi

echo "== 2. Vento misurato: METAR dell'aeroporto di Genova (LIMJ)"
OUT="$DEST/metar_LIMJ_${INIZIO}_${FINE}.csv"
if [ -s "$OUT" ]; then
  echo "  gia' presente: $(basename "$OUT")"
else
  # la data finale del servizio e' esclusa: si chiede fino al giorno dopo
  DOPO=$(python3 -c "import datetime as d; print(d.date.fromisoformat('$FINE') + d.timedelta(days=1))")
  URL="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=LIMJ&data=all&tz=Etc/UTC&format=comma&latlon=yes"
  URL="$URL&year1=${INIZIO:0:4}&month1=${INIZIO:5:2}&day1=${INIZIO:8:2}&year2=${DOPO:0:4}&month2=${DOPO:5:2}&day2=${DOPO:8:2}"
  curl -sSfL --retry 5 --retry-delay 15 -o "$OUT" "$URL"
  echo "  $(basename "$OUT"): $(grep -vc '^#' "$OUT") righe (una ogni 30 minuti circa)"
fi

echo
echo "Fatto. File in $DEST:"
ls -lh "$DEST"
