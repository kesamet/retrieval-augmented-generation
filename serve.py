import logging
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from src.retrieval_qa import load_retrieval_qa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Loading DB and model")
retrieval_qa = load_retrieval_qa()
logging.info("DB and model loaded")


class Request(BaseModel):
    user_query: str
    max_new_tokens: int
    temperature: float
    topk: int


def query(payload: dict) -> Any:
    # TODO
    response = retrieval_qa({'query': payload["user_query"]})
    return response


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Retrieval augmented generation",
        version="1.0.0",
        description="Retrieval augmented generation API",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI()
app.openapi = custom_openapi


@app.post("/")
async def get_response(request: Request) -> Any:
    return query(request.dict())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
