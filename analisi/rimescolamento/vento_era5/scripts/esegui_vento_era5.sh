#!/usr/bin/env bash
# Sturla 2026: le stesse run con il vento ERA5 al posto del vento CMEMS.
# Non sovrascrive le run precedenti: ogni run va in cases/VENTO_<data>_<caso>.
#
#   bash scripts/esegui_vento_era5.sh check      controlli, senza lanciare GOTM
#   bash scripts/esegui_vento_era5.sh all        tutti gli 8 casi
#   bash scripts/esegui_vento_era5.sh NOME_CASO  un caso solo
#   bash scripts/esegui_vento_era5.sh pacchetto  risultati_vento_era5.zip con l'ultima run di ogni caso
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOTM_BIN="${GOTM_BIN:-$HOME/gotm/code/build-sturla-cvmix-debug/gotm}"
CASES=(OTT04_R0_CMEMS OTT04_R0_ERA5 OTT04_R3_CMEMS OTT04_R3_ERA5
       OTTOBRE18_R0_ERA5 OTTOBRE18_R3_ERA5 AGOSTO_R0_ERA5 AGOSTO_R3_ERA5)

preflight() {
  python3 -I "$ROOT/scripts/controlla_vento_era5.py" "$ROOT" "${CASES[@]}"
  if [[ ! -x "$GOTM_BIN" ]]; then
    echo "ERRORE: GOTM con patch CVMix non trovato: $GOTM_BIN" >&2
    echo "Imposta GOTM_BIN=/percorso/effettivo/gotm bash scripts/esegui_vento_era5.sh ..." >&2
    return 10
  fi
  echo "BINARIO selezionato: $GOTM_BIN (serve quello con la patch CVMix: la convezione e' accesa)"
}

one() {
  local name="$1" stamp out
  stamp="$(date -u +%Y%m%d_%H%M%S_%N)"
  out="$ROOT/cases/VENTO_${stamp}_${name}"
  [[ -f "$ROOT/templates/$name/gotm.yaml" ]] || { echo "ERRORE: caso sconosciuto $name" >&2; return 11; }
  [[ ! -e "$out" ]] || { echo "ERRORE: esiste $out" >&2; return 12; }
  mkdir -p "$ROOT/cases"
  cp -a "$ROOT/templates/$name" "$out"
  {
    echo "CASO=$name"
    echo "DATA_CREAZIONE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "BINARIO_GOTM=$GOTM_BIN"
    sha256sum "$GOTM_BIN" "$out/gotm.yaml"
  } > "$out/PROVENIENZA_RUN.txt"
  echo "--- ESEGUO $name ---"
  ulimit -c 0 || true
  if (cd "$out" && "$GOTM_BIN" gotm.yaml >run.log 2>&1); then
    if [[ -s "$out/sturla.nc" ]] && grep -q 'GOTM finished' "$out/run.log"; then
      echo "OK $name: $out/sturla.nc"
    else
      echo "ERRORE: $name exit=0 ma NetCDF/log finale mancanti: $out" >&2
      tail -n 60 "$out/run.log" >&2
      return 13
    fi
  else
    echo "ERRORE $name: $out/run.log" >&2
    tail -n 70 "$out/run.log" >&2
    return 14
  fi
}

pacchetto() {
  python3 -I - "$ROOT" "${CASES[@]}" <<'PY'
import sys, zipfile
from pathlib import Path
root = Path(sys.argv[1]); out = root / "risultati_vento_era5.zip"; n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sys.argv[2:]:
        runs = sorted(root.glob(f"cases/VENTO_*_{name}"))
        runs = [r for r in runs if r.name.split("_", 4)[-1] == name and (r / "sturla.nc").is_file()]
        if not runs:
            print(f"manca una run completa di {name}"); continue
        r = runs[-1]
        for f in ("sturla.nc", "gotm.yaml", "run.log", "PROVENIENZA_RUN.txt"):
            if (r / f).is_file():
                z.write(r / f, f"vento_era5/{r.name}/{f}")
        n += 1
print(f"scritto {out} con {n} run")
PY
}

main() {
  local cmd="${1:-check}" x status=0
  case "$cmd" in
    check) preflight ;;
    all) preflight; for x in "${CASES[@]}"; do one "$x" || status=1; done ;;
    pacchetto) pacchetto ;;
    *) for x in "${CASES[@]}"; do
         if [[ "$x" == "$cmd" ]]; then preflight; one "$cmd"; return $?; fi
       done
       echo 'Uso: bash scripts/esegui_vento_era5.sh check|all|pacchetto|NOME_CASO' >&2
       echo "Casi: ${CASES[*]}" >&2
       return 2 ;;
  esac
  return "$status"
}
main "$@"
