"""
main.py — FastAPI Application & Endpoints
==============================================================================

WHY FastAPI:
    Async-capable, auto-generates OpenAPI docs, native Pydantic v2
    integration, and excellent performance via Starlette + Uvicorn.

STARTUP STRATEGY:
    The lifespan context manager loads the pickled graph and initialises
    the vector engine / recommender *once* on startup, keeping them in
    app.state for the lifetime of the process.  This avoids re-loading
    the ~22 M parameter embedding model on every request.

KNOWN LIMITATIONS:
    - Single-process; no horizontal scaling without shared state.
      Acceptable for a portfolio demo.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os

from config import settings
from models import (
    RecommendRequest,
    RecommendResponse,
    ErrorResponse,
)
from data_manager import TMDBClient, SQLiteCache
from vector_engine import VectorEngine
from graph_engine import load_graph
from recommender import HybridRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load heavy resources once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load graph, vector engine, and recommender into app.state."""
    logger.info("🚀 Starting up — loading resources …")

    # SQLite cache
    db = SQLiteCache()
    app.state.db = db

    # ChromaDB + sentence-transformer
    vector_engine = VectorEngine()
    app.state.vector_engine = vector_engine

    # NetworkX graph (pickled on disk by setup.py)
    try:
        graph = load_graph()
    except FileNotFoundError:
        logger.warning(
            "Graph pickle not found at '%s'. Run setup.py first. "
            "Starting with an empty graph.",
            settings.graph_pickle_path,
        )
        import networkx as nx
        graph = nx.Graph()
    app.state.graph = graph

    # TMDB client (for cold-start fallback)
    try:
        tmdb_client = TMDBClient()
    except ValueError:
        logger.warning("TMDB_API_KEY not set — cold-start disabled.")
        tmdb_client = None
    app.state.tmdb_client = tmdb_client

    # Hybrid recommender
    app.state.recommender = HybridRecommender(
        vector_engine=vector_engine,
        graph=graph,
        db=db,
        tmdb_client=tmdb_client,
    )

    logger.info(
        "✅ Ready — %d movies indexed, %d graph nodes.",
        vector_engine.count(),
        graph.number_of_nodes(),
    )
    yield

    # Cleanup
    db.close()
    logger.info("🛑 Shutdown complete.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hybrid Movie Recommender",
    description=(
        "Combines semantic vector search (all-MiniLM-L6-v2 + ChromaDB) "
        "with a knowledge graph (NetworkX) for explainable movie recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", summary="Serve the UI", include_in_schema=False)
async def root():
    """Serve the single-page application frontend."""
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Seed movie not found."},
    },
    summary="Get hybrid movie recommendations",
)
async def recommend(body: RecommendRequest):
    """Return top-N recommendations for a seed movie.

    Accepts either a movie title (fuzzy-matched via vector search)
    or an exact TMDB ID.  The response includes full XAI provenance
    for every recommendation.
    """
    try:
        result = app.state.recommender.recommend(
            seed_title=body.title,
            seed_tmdb_id=body.tmdb_id,
            top_n=settings.top_n_results,
        )
        return RecommendResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="seed_not_found",
                detail=str(exc),
            ).model_dump(),
        )


@app.get("/health", summary="Health check")
async def health():
    """Return API status, indexed movie count, and graph node count."""
    return {
        "status": "healthy",
        "movies_indexed": app.state.vector_engine.count(),
        "graph_nodes": app.state.graph.number_of_nodes(),
        "graph_edges": app.state.graph.number_of_edges(),
    }


@app.get(
    "/movie/{tmdb_id}",
    summary="Get raw movie metadata",
    responses={404: {"model": ErrorResponse}},
)
async def get_movie(tmdb_id: int):
    """Return raw movie metadata from SQLite (debugging endpoint)."""
    movie = app.state.db.get_movie(tmdb_id)
    if not movie:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="movie_not_found",
                detail=f"No movie with tmdb_id={tmdb_id} in the local cache.",
            ).model_dump(),
        )
    return movie


@app.get("/search", summary="Search movies for autocomplete")
def search_movies(q: str):
    """Search for movies by title substring."""
    if not q:
        return []
    movies = app.state.db.search_movies_by_title(q, limit=8)
    return [{"tmdb_id": m["tmdb_id"], "title": m["title"]} for m in movies]


@app.get("/poster/{tmdb_id}", summary="Get movie poster URL from TMDB")
def get_poster(tmdb_id: int):
    """Fetch the poster_path directly from TMDB."""
    if not app.state.tmdb_client:
        return {"poster_url": None}
    try:
        data = app.state.tmdb_client._get(f"/movie/{tmdb_id}")
        if data.get("poster_path"):
            return {"poster_url": f"https://image.tmdb.org/t/p/w500{data['poster_path']}"}
        return {"poster_url": None}
    except Exception:
        return {"poster_url": None}
