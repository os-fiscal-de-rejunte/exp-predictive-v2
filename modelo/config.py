from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARTEFATO = RAIZ / "modelo" / "artefatos" / "modelo.joblib"
RELATORIO = RAIZ / "modelo" / "artefatos" / "relatorio.json"

NUMERICAS = [
    "grid_lat", "grid_lon", "hotspot_count", "total_frp_mw",
    "max_frp_mw", "temp_2m_c", "temp_2m_max_c",
    "rel_humidity_2m_pct", "precip_mm_day", "wind_speed_10m_ms",
    "soil_moisture_top", "soil_moisture_root", "dia_seno", "dia_cosseno",
]
CATEGORICAS = ["country"]
ENTRADAS = NUMERICAS + CATEGORICAS
ALVO = "hotspot_dia_seguinte"
