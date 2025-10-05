import requests
import pandas as pd

BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


MISSION_TABLES = {
    "kepler": "q1_q17_dr25_koi",
    "k2": "k2pandc",
    "tess": "toi" 
}

def fetch_planet_data(mission: str, planet_id: str) -> pd.DataFrame:
    """
    Fetch full exoplanet observation data for a given mission by planet ID.

    Args:
        mission (str): 'kepler', 'k2', or 'tess'
        planet_id (str): Planet name or ID

    Returns:
        pd.DataFrame: Full observation data
    """
    mission = mission.lower()
    if mission not in MISSION_TABLES:
        raise ValueError(f"Unknown mission '{mission}'. Choose from {list(MISSION_TABLES.keys())}")

    table = MISSION_TABLES[mission]

    if mission == "kepler":
        # Only compare string name columns
        query = f"SELECT * FROM {table} WHERE kepler_name='{planet_id}' OR kepoi_name='{planet_id}'"
    elif mission == "k2":
        query = f"SELECT * FROM {table} WHERE pl_name='{planet_id}' OR tic_id='{planet_id}'"
    elif mission == "tess":
        query = f"SELECT * FROM {table} WHERE toi='{planet_id}' OR tid='{planet_id}'"

    response = requests.get(BASE_URL, params={"query": query, "format": "json"})
    response.raise_for_status()
    return pd.DataFrame(response.json())


def search_planets_by_name(mission: str, name_query: str) -> pd.DataFrame:
    """
    Search planets by partial name for a given mission.

    Args:
        mission (str): 'kepler', 'k2', or 'tess'
        name_query (str): Partial planet name to search

    Returns:
        pd.DataFrame: Matching planet entries
    """
    mission = mission.lower()
    if mission not in MISSION_TABLES:
        raise ValueError(f"Unknown mission '{mission}'. Choose from {list(MISSION_TABLES.keys())}")

    table = MISSION_TABLES[mission]

    if mission == "kepler":
        query = f"SELECT TOP 5 kepler_name, kepoi_name FROM {table} WHERE kepler_name LIKE '%{name_query}%' OR kepoi_name LIKE '%{name_query}%'"
    elif mission == "k2":
        query = f"SELECT TOP 5 pl_name, k2_name FROM {table} WHERE pl_name LIKE '%{name_query}%' OR k2_name LIKE '%{name_query}%'"
    elif mission == "tess":
        query = f"SELECT TOP 5 toi, tid FROM {table} WHERE toi LIKE '%{name_query}%' OR tid LIKE '%{name_query}%'"

    response = requests.get(BASE_URL, params={"query": query, "format": "json"})
    response.raise_for_status()
    return pd.DataFrame(response.json())



# Example usage
if __name__ == "__main__":
    # Fetch Kepler planet
    kepler_df = fetch_planet_data("kepler", "Kepler-22 b")
    print(kepler_df.head())

    # Search K2 planets
    k2_df = search_planets_by_name("k2", "K2-18")
    print(k2_df.head())

    # Fetch TESS planet
    tess_df = fetch_planet_data("tess", "50365310")
    print(tess_df.head())
