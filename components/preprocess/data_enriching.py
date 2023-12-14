import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.external.sncb_operating_points import get_operational_points
from src.framework import Component


class DataEnrichingComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        - InfraBEL Open Data to find the closest station for each point.
        - TODO: Open-Meteo API to get the weather data for each point.

        :param source: The dataframe containing the points
        :return: The dataframe with the new column
        """
        operational_points_gdf = get_operational_points()

        # Convert source to a GeoDataFrame
        geometry = [Point(lat, lon) for lon, lat in zip(source["lon"], source["lat"])]
        geo_source = gpd.GeoDataFrame(source, geometry=geometry)
        geo_source.crs = "EPSG:4326"
        geo_source.to_crs(epsg=3857, inplace=True)

        # Use GeoPandas spatial join to find the closest station for each point in source
        merged = gpd.sjoin_nearest(
            geo_source, operational_points_gdf, how="left", distance_col="geo_distance"
        )

        # Add necessary columns from the result
        result_df = source.copy()
        result_df["nearest_stop"] = merged["geo_point_2d"]
        result_df["stop_type"] = merged["classification"]
        result_df["stop_name"] = merged["commerciallongnamefrench"]
        result_df["stop_distance"] = merged["geo_distance"]
        return result_df
