"""Production-shaped FastAPI service backed by PostgreSQL/PostGIS."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .postgis_repository import PostGISRepository

repo = PostGISRepository()

@asynccontextmanager
async def lifespan(_app):
    repo.open()
    yield
    repo.close()

app = FastAPI(title="NSU Navigation API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Point(BaseModel):
    lat: float
    lon: float

class RouteRequest(BaseModel):
    campus_id: str
    start: Point
    destination: Point

@app.get("/health")
def health():
    try:
        return {"status":"ok", "storage":"postgis", "campus_count":len(repo.campuses())}
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {exc.__class__.__name__}")

@app.get("/api/campuses")
def campuses():
    return repo.campuses()

@app.get("/api/campuses/{campus_id}")
def campus(campus_id: str):
    info = repo.graph_info(campus_id)
    if not info: raise HTTPException(404, "published campus graph not found")
    return {"campus_id":campus_id, **info}

@app.post("/api/routes")
def route(request: RouteRequest):
    result = repo.route(request.campus_id, request.start.model_dump(), request.destination.model_dump())
    if not result: raise HTTPException(422, "no connected route")
    return {"campus_id":request.campus_id, **result}
