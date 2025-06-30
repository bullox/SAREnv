# sarenv/utils/geo.py
"""
Geographic utility functions.
"""
def get_utm_epsg(lon: float, lat: float) -> str:
    """Calculates the appropriate UTM zone EPSG code for a given point."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{326 if lat >= 0 else 327}{zone}"