import logging
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from src.framework import load_config_from_file, StorageManager
from src.framework.outliers import (
    get_ranking_for_components,
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
async def get_ranking(selection: Optional[List[str]] = None):
    # This function should interact with your data source to retrieve rankings.
    # For demonstration, I'm returning a mock response.
    component_names = list(available_components) if not selection else selection
    components = [get_component_from_names(name) for name in component_names]

    progressive_ranking = get_ranking_for_components(components, storage_manager)

    # Transform df into a list of dicts
    progressive_ranking = progressive_ranking.reset_index()

    progressive_ranking = progressive_ranking.rename(columns={"index": "train_id", 0: "ranking"})

    output = [
        {
            "train_id": row["train_id"],
            "ranking": row["ranking"],
        }
        for _, row in progressive_ranking.iterrows()
    ]

    return output