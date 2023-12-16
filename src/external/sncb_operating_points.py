from functools import lru_cache

import geopandas as gpd
import requests


@lru_cache(maxsize=None)
def get_operational_points() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame.from_features(
        requests.get(
            "https://api.mobilitytwin.brussels/infrabel/operational-points",
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer 42227799ae2e74ebc42ca66dee38f4352456c2e93a21962133e0056fd228392eecd70222df0a0c3882438acdfb59de933c50ef368cebb8f5ab8b19d3bd8d2134",
            },
        ).json()["features"],
    ).set_crs("EPSG:3857")[["geometry", "classification"]]
