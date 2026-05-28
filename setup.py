"""
setup.py — One-Time Data Pipeline: Fetch → Cache → Embed → Graph
==============================================================================

Run this script ONCE before starting the API:

    python setup.py

It is designed to be **idempotent** — running it again skips movies
that have already been fetched and embedded.

PIPELINE STAGES:
    1. Fetch top N movies (by popularity) from TMDB, enriching each
       with credits and keywords.  Cache every response in SQLite.
    2. Embed all un-embedded synopses into ChromaDB.
    3. Build (or rebuild) the NetworkX knowledge graph and pickle it.

ESTIMATED RUNTIME (1 000 movies):
    ~25–40 minutes on a residential connection, dominated by TMDB API
    latency (~3 requests per movie × rate limit pauses).

KNOWN LIMITATIONS:
    - TMDB's /discover endpoint returns at most 500 pages (10 000
      movies).  For 1 000 movies we need only 50 pages.
    - Some movies lack an English overview; they are indexed with an
      empty synopsis and skipped during embedding.
"""

import sys
import math
import time
import logging

from config import settings
from data_manager import TMDBClient, SQLiteCache
from vector_engine import VectorEngine
from graph_engine import build_graph, save_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

MOVIES_PER_PAGE = 20  # TMDB returns 20 results per page


def main() -> None:
    """Execute the full setup pipeline."""
    settings.ensure_directories()

    tmdb = TMDBClient()
    db = SQLiteCache()

    total_target = settings.setup_movie_count
    total_pages = math.ceil(total_target / MOVIES_PER_PAGE)

    # =================================================================
    # Stage 1: Fetch movies from TMDB
    # =================================================================
    logger.info(
        "═══ STAGE 1/3: Fetching %d movies from TMDB (%d pages) ═══",
        total_target,
        total_pages,
    )

    fetched = 0
    skipped = 0

    for page in range(1, total_pages + 1):
        logger.info("  Fetching page %d / %d …", page, total_pages)

        try:
            movies = tmdb.get_popular_movies(page=page)
        except Exception as e:
            logger.error("  Failed to fetch page %d: %s", page, e)
            continue

        for movie_stub in movies:
            if fetched >= total_target:
                break

            tmdb_id = movie_stub["id"]

            # Idempotency: skip if already cached
            if db.movie_exists(tmdb_id):
                skipped += 1
                fetched += 1
                continue

            # Enrich with credits and keywords
            try:
                credits = tmdb.get_movie_credits(tmdb_id)
                keywords = tmdb.get_movie_keywords(tmdb_id)
            except Exception as e:
                logger.warning(
                    "  Skipping movie %d — credits/keywords fetch failed: %s",
                    tmdb_id,
                    e,
                )
                continue

            movie_record = {
                "tmdb_id": tmdb_id,
                "title": movie_stub.get("title", ""),
                "overview": movie_stub.get("overview", ""),
                "director": credits["director"],
                "cast": credits["cast"],
                "genres": [
                    g["name"]
                    for g in movie_stub.get("genre_ids", [])
                ]
                if isinstance(movie_stub.get("genre_ids"), list)
                and movie_stub.get("genre_ids")
                and isinstance(movie_stub["genre_ids"][0], dict)
                else [],
                "keywords": keywords,
                "embedded": False,
            }

            # TMDB discover returns genre_ids as ints, not dicts.
            # We'll store empty genres for now; they'll be available
            # in metadata from the TMDB response's other fields.
            # For a cleaner approach, we could call /movie/{id} for
            # full genre names, but that's another API hit per movie.

            db.save_movie(movie_record)
            fetched += 1

            if fetched % 50 == 0:
                logger.info("  Progress: %d / %d movies fetched.", fetched, total_target)

        if fetched >= total_target:
            break

    logger.info(
        "  ✅ Fetching complete: %d total (%d new, %d already cached).",
        fetched,
        fetched - skipped,
        skipped,
    )

    # =================================================================
    # Stage 2: Embed synopses into ChromaDB
    # =================================================================
    logger.info("═══ STAGE 2/3: Embedding synopses into ChromaDB ═══")

    vector_engine = VectorEngine()
    unembedded = db.get_unembedded_movies()
    logger.info("  %d movies need embedding.", len(unembedded))

    for i, movie in enumerate(unembedded, 1):
        vector_engine.index_movie(
            tmdb_id=movie["tmdb_id"],
            title=movie["title"],
            synopsis=movie.get("overview", ""),
            metadata={
                "director": movie.get("director", ""),
                "cast": movie.get("cast", []),
                "genres": movie.get("genres", []),
            },
        )
        db.mark_embedded(movie["tmdb_id"])

        if i % 50 == 0 or i == len(unembedded):
            logger.info("  Embedded %d / %d", i, len(unembedded))

    logger.info(
        "  ✅ Embedding complete: %d total documents in ChromaDB.",
        vector_engine.count(),
    )

    # =================================================================
    # Stage 3: Build the Knowledge Graph
    # =================================================================
    logger.info("═══ STAGE 3/3: Building Knowledge Graph ═══")

    all_movies = db.get_all_movies()
    graph = build_graph(all_movies)
    save_graph(graph)

    logger.info(
        "  ✅ Graph complete: %d nodes, %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    # =================================================================
    # Summary
    # =================================================================
    logger.info("═══ SETUP COMPLETE ═══")
    logger.info("  Movies cached in SQLite : %d", len(all_movies))
    logger.info("  Vectors in ChromaDB     : %d", vector_engine.count())
    logger.info("  Graph nodes             : %d", graph.number_of_nodes())
    logger.info("  Graph edges             : %d", graph.number_of_edges())
    logger.info("")
    logger.info("Start the API with:  uvicorn main:app --reload")

    db.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted. Re-run to continue from where you left off.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Setup failed: %s", e)
        sys.exit(1)
