import geopandas as gpd
import pandas as pd

from src.external.openweather import get_weather_for_period
from src.external.sncb_operating_points import get_operational_points
from src.framework import Component
from src.mining.outliers_detection import train_stopped


class DataEnrichingComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        - InfraBEL Open Data to find the closest station for each point.
        - Open-Meteo API to get the weather data for each point.

        :param source: The dataframe containing the points
        :return: The dataframe with the new column
        """
        operational_points_gdf = get_operational_points()
        weather_data_gdf = get_weather_for_period((source.index[0], source.index[-1]))

        # Convert source to a GeoDataFrame
        source = gpd.GeoDataFrame(
            source, geometry=gpd.points_from_xy(source.lon, source.lat), crs="EPSG:3857"
        )

        # Use GeoPandas spatial join to find the closest station for each point in source
        merged = gpd.sjoin_nearest(
            source,
            operational_points_gdf,
            how="left",
            distance_col="stop_distance",
        )

        # Sort by stop_distance
        merged.sort_values(by="stop_distance", inplace=True)
        # Drop duplicates (keep the first one)
        merged.drop_duplicates(subset=["train_id", "lat", "lon"], inplace=True, keep="first")

        merged = gpd.sjoin_nearest(
            merged, weather_data_gdf, how="left", rsuffix="_weather", distance_col="weather_distance"
        )
        # Sort by weather_distance
        merged.sort_values(by="weather_distance", inplace=True)
        # Drop duplicates (keep the first one)
        merged.drop_duplicates(subset=["train_id", "lat", "lon"], inplace=True, keep="first")

        # Drop geometry columns
        merged.drop(columns=["geometry", "weather_distance"], inplace=True)

        # Now label with at_station and at_stop_point
        merged = train_stopped(merged, threshold=500)

        # Make stop_distance int16, temperature int8, humidity int8
        merged["stop_distance"] = merged["stop_distance"].astype("int16")
        merged["temperature"] = merged["temperature"].astype("int8")
        merged["relative_humidity"] = merged["relative_humidity"].astype("int8")

        merged.sort_index(inplace=True)

        return merged
