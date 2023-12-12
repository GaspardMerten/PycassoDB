import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

from src.framework import Component


class DataEnrichingComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        - InfraBEL Open Data to find the closest station for each point.
        - TODO: Open-Meteo API to get the weather data for each point.

        :param source: The dataframe containing the points
        :return: The dataframe with the new column
        """
        operational_points = gpd.read_file('backup/operating_points.csv', sep=';')
        operational_points = operational_points[
            (operational_points['Classification EN'] == 'Station') |
            (operational_points['Classification EN'] == 'Stop in open track')
        ]

        # Convert 'Geo Point' column in operational_points to a Point geometry
        operational_points['geometry'] = operational_points['Geo Point'].apply(
            lambda x: Point(map(float, x.split(', ')))
        )
        operational_points.crs = "EPSG:4326"
        operational_points.to_crs(epsg=3857, inplace=True)

        # Convert source to a GeoDataFrame
        geometry = [Point(lat, lon) for lon, lat in zip(source['lon'], source['lat'])]
        geo_source = gpd.GeoDataFrame(source, geometry=geometry)
        geo_source.crs = "EPSG:4326"
        geo_source.to_crs(epsg=3857, inplace=True)

        # Use GeoPandas spatial join to find the closest station for each point in source
        merged = gpd.sjoin_nearest(geo_source, operational_points, how='left', distance_col='geo_distance')

        # Add necessary columns from the result
        result_df = source.copy()
        result_df['nearest_stop'] = merged['Nom FR complet']
        result_df['stop_type'] = merged['Classification EN']
        result_df['stop_location'] = merged['Geo Point']
        result_df['stop_distance'] = merged['geo_distance']

        return result_df
