"""
data_manager.py — TMDB API Client & SQLite Caching Layer
==============================================================================

WHY TWO CLASSES (TMDBClient + SQLiteCache):
    Separating the *fetcher* from the *cache* follows the Single
    Responsibility Principle and makes each independently testable.
    The cache can be swapped for Postgres or Redis without touching
    the fetching logic, and the fetcher can be replaced with a mock
    for unit tests.

WHAT PROBLEM IT SOLVES:
    1. Rate-limit–safe TMDB interaction with automatic exponential backoff.
    2. Persistent SQLite cache so that re-running setup.py never wastes
       API calls on already-fetched movies.
    3. Structured storage of complex fields (cast, genres, keywords) as
       JSON strings inside SQLite text columns.

KNOWN LIMITATIONS:
    - SQLite is single-writer; concurrent writes from multiple processes
      will block.  Acceptable for a single-node portfolio project.
    - The retry logic handles HTTP 429 only; other transient errors (5xx)
      are not retried.
"""

import json
import time
import sqlite3
import logging
from typing import Dict, List, Optional

import requests

from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# TMDB API Client
# =============================================================================

class TMDBClient:
    """Thin wrapper around the TMDB REST API.

    WHY a dedicated class instead of raw requests calls:
        • Centralises the API key header and base URL.
        • Encapsulates retry / back-off logic in one place.
        • Makes mocking trivial in tests (patch the class, not requests).
    """

    BASE_URL = "https://api.themoviedb.org/3"

    # ------------------------------------------------------------------
    # WHY max_retries=3 and base_delay=1.0:
    #   TMDB imposes ~40 req/10s.  Three retries with 1→2→4 s delays
    #   covers a single burst without hammering the API.
    # ------------------------------------------------------------------
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds — doubles on each retry

    def __init__(self, api_key: Optional[str] = None):
        """Initialise with an API key (falls back to settings)."""
        self.api_key = api_key or settings.tmdb_api_key
        if not self.api_key:
            raise ValueError(
                "TMDB_API_KEY is not set.  Copy .env.example → .env and add your key."
            )
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key}  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal: resilient HTTP GET
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Issue a GET with automatic retry on 429 (rate limit).

        WHY exponential backoff:
            Linear waits waste time on transient blips and don't adapt
            to sustained pressure.  Exponential backoff is the industry
            standard for rate-limited APIs.
        """
        url = f"{self.BASE_URL}{endpoint}"
        for attempt in range(1, self.MAX_RETRIES + 1):
            resp = self.session.get(url, params=params, timeout=15)

            if resp.status_code == 429:
                # --------------------------------------------------------
                # TMDB returns a Retry-After header (seconds).  We honour
                # it if present; otherwise fall back to our own backoff.
                # --------------------------------------------------------
                wait = float(resp.headers.get("Retry-After", self.BASE_DELAY * (2 ** (attempt - 1))))
                logger.warning(
                    "TMDB rate limit hit (attempt %d/%d). Waiting %.1f s …",
                    attempt,
                    self.MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        # If we exhaust retries, raise the last response's error
        resp.raise_for_status()
        return {}  # unreachable, but satisfies type checkers

    # ------------------------------------------------------------------
    # Public: TMDB data accessors
    # ------------------------------------------------------------------

    def get_popular_movies(self, page: int = 1) -> List[Dict]:
        """Fetch one page of popular movies via /discover/movie.

        WHY /discover instead of /movie/popular:
            /discover supports sorting by popularity.desc *and*
            returns up to 500 pages (10 000 movies), whereas
            /movie/popular caps out much sooner.
        """
        data = self._get(
            "/discover/movie",
            params={
                "sort_by": "popularity.desc",
                "page": page,
                "language": "en-US",
                "include_adult": "false",
            },
        )
        return data.get("results", [])

    def get_movie_credits(self, tmdb_id: int) -> Dict:
        """Extract the director and top-3 billed cast from /credits.

        WHY top-3 only:
            Limiting cast to the 3 most prominent actors keeps the
            knowledge graph sparse enough for meaningful edges while
            avoiding noise from bit-part players who appear in dozens
            of unrelated films.

        Returns:
            {"director": str, "cast": [str, str, str]}
        """
        data = self._get(f"/movie/{tmdb_id}/credits")

        # Director — first person in crew with job == "Director"
        director = ""
        for member in data.get("crew", []):
            if member.get("job") == "Director":
                director = member.get("name", "")
                break

        # Top 3 billed cast — /credits returns cast already sorted by "order"
        cast = [
            member.get("name", "")
            for member in data.get("cast", [])[:3]
        ]

        return {"director": director, "cast": cast}

    def get_movie_keywords(self, tmdb_id: int) -> List[str]:
        """Fetch keyword strings for a movie.

        Keywords are used as extra metadata in ChromaDB but do NOT
        influence graph edges (which rely solely on director / cast).
        """
        data = self._get(f"/movie/{tmdb_id}/keywords")
        return [kw.get("name", "") for kw in data.get("keywords", [])]

    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch full movie details (title, overview, genres) by ID.

        Used by the cold-start handler when a user queries a movie that
        was never part of the initial setup batch.
        """
        try:
            data = self._get(f"/movie/{tmdb_id}")
            return {
                "tmdb_id": data["id"],
                "title": data.get("title", ""),
                "overview": data.get("overview", ""),
                "genres": [g["name"] for g in data.get("genres", [])],
            }
        except requests.HTTPError:
            return None


# =============================================================================
# SQLite Cache
# =============================================================================

class SQLiteCache:
    """Persistent movie cache backed by SQLite.

    WHY SQLite over a plain JSON file:
        • Atomic writes — no corruption on crash.
        • Indexed lookups by tmdb_id are O(log n) vs O(n) for JSON scan.
        • SQL queries are trivially extensible (e.g. "give me all un-embedded movies").

    WHY a single table:
        At portfolio scale (~1 000 movies) normalisation into separate
        tables for cast / genres / keywords adds complexity with zero
        performance benefit.  Denormalised JSON columns are the pragmatic choice.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS movies (
        tmdb_id   INTEGER PRIMARY KEY,
        title     TEXT    NOT NULL,
        overview  TEXT    NOT NULL DEFAULT '',
        director  TEXT    NOT NULL DEFAULT '',
        cast      TEXT    NOT NULL DEFAULT '[]',   -- JSON array of top-3 names
        genres    TEXT    NOT NULL DEFAULT '[]',   -- JSON array of genre strings
        keywords  TEXT    NOT NULL DEFAULT '[]',   -- JSON array of keyword strings
        embedded  INTEGER NOT NULL DEFAULT 0       -- 0 = not yet embedded
    );
    """

    def __init__(self, db_path: Optional[str] = None):
        """Open (or create) the SQLite database at `db_path`."""
        self.db_path = db_path or settings.sqlite_db_path
        # ------------------------------------------------------------------
        # WHY check_same_thread=False:
        #   FastAPI runs request handlers in a thread pool.  SQLite's
        #   default single-thread enforcement would raise errors.
        #   We serialise writes via the GIL, which is safe enough here.
        # ------------------------------------------------------------------
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(self.DDL)
        self.conn.commit()

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def save_movie(self, movie: Dict) -> None:
        """Insert or update a movie record.

        Uses INSERT OR REPLACE so re-running setup with the same data
        is idempotent — no duplicate-key errors.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO movies
                (tmdb_id, title, overview, director, cast, genres, keywords, embedded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                movie["tmdb_id"],
                movie["title"],
                movie.get("overview", ""),
                movie.get("director", ""),
                json.dumps(movie.get("cast", [])),
                json.dumps(movie.get("genres", [])),
                json.dumps(movie.get("keywords", [])),
                int(movie.get("embedded", False)),
            ),
        )
        self.conn.commit()

    def get_movie(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch a single movie by TMDB ID, or None if not cached."""
        row = self.conn.execute(
            "SELECT * FROM movies WHERE tmdb_id = ?", (tmdb_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_movies(self) -> List[Dict]:
        """Return every cached movie as a list of dicts."""
        rows = self.conn.execute("SELECT * FROM movies").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search_movies_by_title(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for movies by title substring for autocomplete."""
        rows = self.conn.execute(
            "SELECT * FROM movies WHERE title LIKE ? LIMIT ?", (f"%{query}%", limit)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def movie_exists(self, tmdb_id: int) -> bool:
        """Fast existence check for idempotency during setup."""
        row = self.conn.execute(
            "SELECT 1 FROM movies WHERE tmdb_id = ? LIMIT 1", (tmdb_id,)
        ).fetchone()
        return row is not None

    def mark_embedded(self, tmdb_id: int) -> None:
        """Flag a movie as having been embedded into ChromaDB."""
        self.conn.execute(
            "UPDATE movies SET embedded = 1 WHERE tmdb_id = ?", (tmdb_id,)
        )
        self.conn.commit()

    def get_unembedded_movies(self) -> List[Dict]:
        """Return movies that have not yet been indexed into ChromaDB.

        WHY this method exists:
            If setup.py crashes mid-embedding, re-running it should
            only process the remaining un-embedded movies — not re-embed
            everything.
        """
        rows = self.conn.execute(
            "SELECT * FROM movies WHERE embedded = 0"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        """Convert a sqlite3.Row into a plain dict with decoded JSON fields."""
        d = dict(row)
        for field in ("cast", "genres", "keywords"):
            if isinstance(d.get(field), str):
                d[field] = json.loads(d[field])
        d["embedded"] = bool(d.get("embedded", 0))
        return d

    def close(self) -> None:
        """Explicitly close the database connection."""
        self.conn.close()
