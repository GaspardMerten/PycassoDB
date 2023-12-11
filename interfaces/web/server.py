from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

# Create a FastAPI instance
app = FastAPI()

import os

current_dir = os.path.dirname(os.path.realpath(__file__))

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))


# Create a route that renders a template
@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": "Hello Ninja Templates with FastAPI!"},
    )
