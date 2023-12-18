import base64
import io
import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from matplotlib import pyplot as plt
from starlette.staticfiles import StaticFiles

from src.framework import load_config_from_file, StorageManager
from src.framework.outliers import (
    get_ranking_for_components,
    get_outliers_for_train,
)

# Create a FastAPI instance
app = FastAPI()

import os

current_dir = os.path.dirname(os.path.realpath(__file__))

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))


app.mount(
    "/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static"
)

logging.basicConfig(level=logging.INFO)

# Load the global and component configuration
config = load_config_from_file("config.toml")
# Instantiate the storage manager, which handles all data storage for the components
storage_manager = StorageManager(config.storage_folder)

available_components = []

for name, component_config in config.components.items():
    if not component_config.outliers_producer:
        continue

    available_components.append(name)


# Create a route that renders a template
@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse(
        "pages/index.html",
        {"request": request, "components": available_components},
    )


def get_component_from_names(name: str):
    return config.components[name]


@app.get("/api/ranking")
async def get_ranking():
    # This function should interact with your data source to retrieve rankings.
    # For demonstration, I'm returning a mock response.

    rankings = []
    for component in config.components.values():
        if not component.outliers_producer:
            continue
        rankings.append(
            dict(
                items=get_ranking_for_components(
                    [component],
                    storage_manager,
                ),
                name=component.name,
            )
        )

    return rankings


@app.get("/api/outliers")
def get_outliers(
    component_name: str,
    train_id: Optional[str] = None,
):
    outliers = get_outliers_for_train(train_id, component_name, storage_manager)

    response = {}

    if "lat" in outliers.columns:
        # Make a beautiful plot with the outliers on a basemap (openstreetmap)
        import contextily as ctx
        import geopandas as gpd

        # Create a GeoDataFrame from the outliers
        outliers = gpd.GeoDataFrame(
            outliers,
            geometry=gpd.points_from_xy(outliers.lon, outliers.lat),
            crs="EPSG:4326",
        )

        outliers.plot(color="red", alpha=0.5, markersize=20)

        # Add a basemap (good quality map) (cover entire belgium)
        ctx.add_basemap(
            plt.gca(),
            crs=outliers.crs.to_string(),
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=12,
        )

        # Save image to base64 in response
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        base64_plot = base64.b64encode(buffer.read()).decode()
        response["map"] = base64_plot
        plt.close()

    # round timestamp to hour (is DateTimeIndex)
    outliers["t"] = outliers.index.round("H")
    # count outliers per hour
    outliers = outliers.groupby("t").count()

    # Plot the evolution of the outliers
    plt.bar(
        x=outliers.index,
        height=outliers[outliers.columns[0]],
    )

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    base64_plot = base64.b64encode(buffer.read()).decode()
    response["evolution"] = base64_plot
    plt.close()

    return response
