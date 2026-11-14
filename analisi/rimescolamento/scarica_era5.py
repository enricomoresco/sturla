"""Scarica da ERA5, nello stesso riquadro dei flussi gia' usati, il vento a 10 m, lo sforzo del vento
in superficie e la pressione al livello del mare (orari).

Servono: pip install cdsapi, un account sul Climate Data Store (cds.climate.copernicus.eu),
l'accettazione della licenza di ERA5 sul sito e il file ~/.cdsapirc con la propria chiave.

  python analisi/rimescolamento/scarica_era5.py [cartella] [anno] [mesi separati da virgola]
  (default: dati_esterni 2026 07,08,09,10)
"""
import os
import sys

import cdsapi


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "dati_esterni"
    anno = sys.argv[2] if len(sys.argv) > 2 else "2026"
    mesi = sys.argv[3].split(",") if len(sys.argv) > 3 else ["07", "08", "09", "10"]
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, f"era5_vento_pressione_sturla_{anno}.nc")
    if os.path.exists(out):
        print(f"gia' presente: {out}")
        return
    richiesta = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure",
                     "mean_eastward_turbulent_surface_stress", "mean_northward_turbulent_surface_stress"],
        "year": [anno],
        "month": [f"{int(m):02d}" for m in mesi],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [44.5, 8.75, 44.0, 9.25],          # nord, ovest, sud, est: come il file dei flussi
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", richiesta, out)
    print(f"scritto {out}")


if __name__ == "__main__":
    main()
