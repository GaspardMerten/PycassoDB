from functools import lru_cache

import geopandas as gpd
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from src.framework import Period

# Set up the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
open_meteo = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"

coordinates = {
    "Hasselt": [50.930965, 5.338333],
    "Brussels": [50.85045, 4.34878],
    "Antwerp": [51.21989, 4.40346],
    "Ghent": [51.05, 3.71667],
    "Charleroi": [50.41136, 4.44448],
}


@lru_cache(maxsize=128)
def get_weather_for_period(period: Period) -> gpd.GeoDataFrame:
    output = pd.DataFrame()
    for city, coordinate in coordinates.items():
        params = {
            # Brussels coordinates
            "latitude": coordinate[0],
            "longitude": coordinate[1],
            "hourly": ["temperature_2m", "relative_humidity_2m"],
            "start_date": period[0].to_pydatetime().date().isoformat(),
            "end_date": period[1].to_pydatetime().date().isoformat(),
        }
        responses = open_meteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature": hourly_temperature_2m,
            "relative_humidity": hourly_relative_humidity_2m,
            "latitude": coordinate[0],
            "longitude": coordinate[1],
        }

        output = pd.concat([output, pd.DataFrame(hourly_data)])

    # Convert to GeoDataFrame
    output = gpd.GeoDataFrame(
        output,
        geometry=gpd.points_from_xy(output.longitude, output.latitude),
        crs="EPSG:4326",
    )

    # Drop latitude and longitude columns
    output.drop(columns=["latitude", "longitude"], inplace=True)

    output.to_crs("EPSG:3857", inplace=True)

    return output
