EVENTOS_QUALITATIVAS = [
    "sensor_family", "satellite", "proc_level", "country", "state", "daynight",
]

EVENTOS_QUANTITATIVAS = [
    "latitude", "longitude", "confidence_norm", "pixel_area_km2",
    "bt_mir_kelvin", "bt_tir_kelvin", "thermal_anomaly_delta_k",
    "normalized_thermal_index", "frp_mw", "frp_density_mw_km2",
    "confidence_weighted_frp",
]

EVENTOS_TEMPORAIS = ["acq_timestamp_utc"]

CELULAS_QUALITATIVAS = [
    "country", "state", "land_cover_class", "has_valid_satellite_obs",
]

CELULAS_QUANTITATIVAS = [
    "grid_lat", "grid_lon", "hotspot_count", "night_event_ratio",
    "total_frp_mw", "mean_frp_mw", "max_frp_mw",
    "mean_frp_density_mw_km2", "max_frp_density_mw_km2",
    "sum_confidence_weighted_frp", "max_thermal_anomaly_delta_k",
    "mean_normalized_thermal_index", "sp_coverage_ratio",
    "frp_density_daily_growth_rate", "rolling_3d_frp_max",
    "rolling_3d_hotspot_sum", "temp_2m_c", "temp_2m_max_c",
    "rel_humidity_2m_pct", "precip_mm_day", "wind_speed_10m_ms",
    "soil_moisture_top", "soil_moisture_root", "modis_valid_obs_ratio",
    "viirs_valid_obs_ratio", "ndvi", "evi", "tree_cover_pct", "elevation_m",
]

CELULAS_TEMPORAIS = ["acq_date"]
CELULAS_IDENTIFICADORES = ["grid_id"]

TABELAS = {
    "curated_harmonized_events": {
        "qualitativas": EVENTOS_QUALITATIVAS,
        "quantitativas": EVENTOS_QUANTITATIVAS,
        "temporais": EVENTOS_TEMPORAIS,
        "identificadores": [],
    },
    "curated_ml_feature_store": {
        "qualitativas": CELULAS_QUALITATIVAS,
        "quantitativas": CELULAS_QUANTITATIVAS,
        "temporais": CELULAS_TEMPORAIS,
        "identificadores": CELULAS_IDENTIFICADORES,
    },
}
