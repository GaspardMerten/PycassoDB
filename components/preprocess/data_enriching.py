import geopandas as gpd
import pandas as pd

from src.external.openweather import get_weather_for_period
from src.external.sncb_operating_points import get_operational_points
from src.framework import Component


class DataEnrichingComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        - InfraBEL Open Data to find the closest station for each point.
        - Open-Meteo API to get the weather data for each point.

        :param source: The dataframe containing the points
        :return: The dataframe with the new column
        """
        print("Enriching data...", source.index.min(), source.index.max())
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

        merged = gpd.sjoin_nearest(
            merged, weather_data_gdf, how="left", rsuffix="_weather"
        )



        # Drop geometry columns
        merged.drop(columns=["geometry"], inplace=True)



        return merged
