"""
recommender.py — Hybrid Re-Ranker with Explainability (XAI)
==============================================================================

WHY HYBRID (Semantic + Graph) BEATS PURE SEMANTIC SEARCH:
    Semantic search finds movies with similar *themes/tone* (e.g. "heist
    movies") but misses factual links like "same director" or "same lead
    actor".  The graph captures those deterministic relationships.
    Combining both signals produces recommendations that are both
    thematically coherent and factually grounded.

SCORING FORMULA:
    final_score = min(cosine_similarity + graph_boost, 1.0)
    This additive approach is simple, interpretable, and effective.

KNOWN LIMITATIONS:
    - The additive formula treats semantic and graph signals as equally
      trustworthy.  A learned weighting (e.g. via a small MLP) would be
      more accurate but adds training complexity.
    - Cold-start movies get partial graph integration (edges only to
      existing nodes).
"""

import logging
from typing import List, Optional

from models import (
    GraphExplanation,
    RecommendationResult,
)
from vector_engine import VectorEngine
from graph_engine import (
    get_graph_boost,
    get_graph_explanation,
    add_movie_to_graph,
    save_graph,
)
from data_manager import TMDBClient, SQLiteCache

import networkx as nx

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Orchestrates the full recommend pipeline: resolve → search → re-rank → explain."""

    def __init__(
        self,
        vector_engine: VectorEngine,
        graph: nx.Graph,
        db: SQLiteCache,
        tmdb_client: Optional[TMDBClient] = None,
    ):
        self.vector_engine = vector_engine
        self.graph = graph
        self.db = db
        self.tmdb = tmdb_client

    def recommend(
        self,
        seed_title: Optional[str] = None,
        seed_tmdb_id: Optional[int] = None,
        top_n: int = 10,
    ) -> dict:
        """Full hybrid recommendation pipeline.

        Returns a dict matching RecommendResponse schema:
            {seed_title, seed_tmdb_id, results: [RecommendationResult, ...]}

        Raises ValueError if the seed movie cannot be resolved.
        """
        # =================================================================
        # Step 1: Resolve the seed movie
        # =================================================================
        seed = self._resolve_seed(seed_title, seed_tmdb_id)
        if seed is None:
            raise ValueError(
                f"Could not resolve seed movie (title={seed_title!r}, "
                f"tmdb_id={seed_tmdb_id})."
            )

        resolved_id = seed["tmdb_id"]
        resolved_title = seed["title"]
        synopsis = seed.get("synopsis") or seed.get("overview", "")

        logger.info("Seed resolved: %s (ID %d)", resolved_title, resolved_id)

        # =================================================================
        # Step 2: Retrieve top-20 semantic candidates
        # =================================================================
        candidates = self.vector_engine.search(synopsis, top_k=20)

        # =================================================================
        # Step 3: Re-rank with graph boost
        # =================================================================
        scored: List[dict] = []
        for c in candidates:
            cid = c["tmdb_id"]
            if cid == resolved_id:
                continue  # exclude the seed itself

            cosine_sim = c["cosine_similarity"]
            g_boost = get_graph_boost(self.graph, resolved_id, cid)
            final = min(cosine_sim + g_boost, 1.0)

            # Build XAI explanation
            graph_conn = get_graph_explanation(self.graph, resolved_id, cid)
            explanation = self._build_explanation(
                cosine_sim, resolved_title, graph_conn
            )

            scored.append({
                "tmdb_id": cid,
                "title": c["title"],
                "cosine_similarity": round(cosine_sim, 4),
                "graph_boost": round(g_boost, 4),
                "final_score": round(final, 4),
                "explanation": explanation,
                "graph_explanation": GraphExplanation(
                    has_connection=bool(graph_conn),
                    connection_detail=graph_conn,
                    graph_boost=round(g_boost, 4),
                ),
            })

        # Sort descending by final_score
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        return {
            "seed_title": resolved_title,
            "seed_tmdb_id": resolved_id,
            "results": [
                RecommendationResult(**r) for r in scored[:top_n]
            ],
        }

    # ------------------------------------------------------------------
    # Seed resolution
    # ------------------------------------------------------------------

    def _resolve_seed(
        self, title: Optional[str], tmdb_id: Optional[int]
    ) -> Optional[dict]:
        """Resolve a seed movie, with cold-start fallback to TMDB.

        Priority: tmdb_id (exact) > title (fuzzy vector search).
        If the movie isn't in ChromaDB, fetch from TMDB, embed, and
        add to the graph on the fly.
        """
        # Try by ID first
        if tmdb_id:
            record = self.vector_engine.get_by_tmdb_id(tmdb_id)
            if record:
                return record
            # Cold-start: fetch from TMDB
            return self._cold_start(tmdb_id)

        # Fuzzy title search via vector engine
        if title:
            results = self.vector_engine.search(title, top_k=1)
            if results:
                return results[0]

        return None

    def _cold_start(self, tmdb_id: int) -> Optional[dict]:
        """Fetch a movie from TMDB, embed it, add to graph.

        WHY cold-start handling:
            Users may query any movie, not just the 1 000 in our setup
            batch.  Fetching on-the-fly and integrating into both the
            vector index and graph means the system degrades gracefully
            rather than returning a hard error.
        """
        if not self.tmdb:
            return None

        logger.info("Cold start: fetching movie %d from TMDB …", tmdb_id)

        details = self.tmdb.get_movie_details(tmdb_id)
        if not details:
            return None

        credits = self.tmdb.get_movie_credits(tmdb_id)
        keywords = self.tmdb.get_movie_keywords(tmdb_id)

        movie = {
            "tmdb_id": details["tmdb_id"],
            "title": details["title"],
            "overview": details["overview"],
            "director": credits["director"],
            "cast": credits["cast"],
            "genres": details["genres"],
            "keywords": keywords,
            "embedded": True,
        }

        # Persist to SQLite
        self.db.save_movie(movie)

        # Index into ChromaDB
        self.vector_engine.index_movie(
            tmdb_id=movie["tmdb_id"],
            title=movie["title"],
            synopsis=movie["overview"],
            metadata={
                "director": movie["director"],
                "cast": movie["cast"],
                "genres": movie["genres"],
            },
        )

        # Add to knowledge graph
        all_movies = self.db.get_all_movies()
        add_movie_to_graph(self.graph, movie, all_movies)
        try:
            save_graph(self.graph)
        except Exception as e:
            logger.warning("Could not persist graph after cold start: %s", e)

        return {
            "tmdb_id": movie["tmdb_id"],
            "title": movie["title"],
            "synopsis": movie["overview"],
            "director": movie["director"],
            "cast": movie["cast"],
            "genres": movie["genres"],
        }

    # ------------------------------------------------------------------
    # XAI explanation builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        cosine_sim: float, seed_title: str, graph_connection: str
    ) -> str:
        """Build a human-readable XAI explanation string.

        Format:
            "Recommended because the synopsis is 82% semantically similar
             to Inception. Also shares director Christopher Nolan with
             the seed movie."
        """
        pct = int(round(cosine_sim * 100))
        base = (
            f"Recommended because the synopsis is {pct}% semantically "
            f"similar to {seed_title}."
        )

        if not graph_connection:
            return base

        # Parse the graph_connection into a friendly suffix
        # graph_connection looks like: "Shares director X and actor Y"
        conn_lower = graph_connection.lower()
        has_director = "director" in conn_lower
        has_actor = "actor" in conn_lower

        if has_director and has_actor:
            # Extract names from the connection string
            suffix = f" Also {graph_connection.lower()} with the seed movie."
        elif has_director:
            suffix = f" Also {graph_connection.lower()} with the seed movie."
        elif has_actor:
            suffix = f" Also {graph_connection.lower()} with the seed movie."
        else:
            suffix = ""

        return base + suffix
