"""
config.py — Centralised, Typed Configuration via python-dotenv
==============================================================================

WHY THIS APPROACH:
    Every configurable value lives in one place, loaded from environment
    variables (or a .env file).  This avoids hard-coded magic strings
    scattered across modules and makes it trivial to swap between dev /
    staging / production settings.

WHAT PROBLEM IT SOLVES:
    1. Secrets (TMDB API key) never appear in source code.
    2. Paths and tunables are overridable without touching Python files.
    3. Type coercion happens once, here — callers get native Python types.

KNOWN LIMITATIONS:
    - No runtime validation beyond basic type casting.  Invalid paths or
      missing keys will surface only when the consuming module tries to
      use them.
"""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root.  `override=False` means real environment
# variables always win — useful in Docker / CI where .env is not mounted.
# ---------------------------------------------------------------------------
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Using a frozen dataclass instead of a plain dict gives us:
      • Attribute access with IDE autocomplete
      • Immutability — prevents accidental mutation at runtime
      • A single, inspectable object that can be passed to any module
    """

    # TMDB API credentials
    tmdb_api_key: str = os.getenv("TMDB_API_KEY", "")

    # Persistence paths (relative paths are resolved from the project root)
    chroma_persist_path: str = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
    graph_pickle_path: str = os.getenv("GRAPH_PICKLE_PATH", "./graph/movie_graph.pkl")
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "./data/movies.db")

    # Tunables
    top_n_results: int = int(os.getenv("TOP_N_RESULTS", "10"))
    setup_movie_count: int = int(os.getenv("SETUP_MOVIE_COUNT", "1000"))

    def ensure_directories(self) -> None:
        """Create parent directories for all persistence paths if missing.

        Called once during setup so that downstream code never has to
        worry about missing folders.
        """
        for path_str in [
            self.chroma_persist_path,
            self.graph_pickle_path,
            self.sqlite_db_path,
        ]:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Module-level singleton.  Import `settings` anywhere to get the
# canonical config object:
#
#     from config import settings
# ---------------------------------------------------------------------------
settings = Settings()
