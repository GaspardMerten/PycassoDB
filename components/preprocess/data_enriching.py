import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

from src.framework import Component


class DataEnrichingComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        This function takes a dataframe as input and adds a column with a boolean value
        indicating whether the point is close to a station or not.

        It uses the data from Infrabel Open Data to find the closest station for each point.

        :param source: The dataframe containing the points
        :return: The dataframe with the new column
        """
        operational_points = gpd.read_file('backup/operating_points.csv', sep=';')

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

        # Add a boolean column for distances below threshold
        threshold = 500  # in meters

        # Add necessary columns from the result
        result_df = source.copy()
        result_df['close_to_station'] = merged['geo_distance'] < threshold

        print(merged['geo_distance'].describe())

        return result_df
