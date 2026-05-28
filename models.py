"""
models.py — Pydantic v2 Request / Response Schemas
==============================================================================

WHY PYDANTIC v2:
    Pydantic v2 (built on pydantic-core in Rust) provides ~5–50× faster
    validation than v1 while keeping a Pythonic API.  Using strict schemas
    here guarantees that every payload entering or leaving the API is
    well-typed, documented via OpenAPI, and self-validating.

WHAT PROBLEM IT SOLVES:
    1. Automatic request validation with descriptive 422 errors.
    2. Response serialisation with guaranteed structure (no stray fields).
    3. Self-documenting API via the generated OpenAPI/Swagger UI.

KNOWN LIMITATIONS:
    - Pydantic validation adds a small per-request overhead (~μs).
      Negligible for this workload.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Input for the /recommend endpoint.

    At least one of `title` or `tmdb_id` must be provided.
    If both are supplied, `tmdb_id` takes precedence because it is
    an unambiguous identifier.
    """

    title: Optional[str] = Field(
        default=None,
        description="Movie title to search for (fuzzy-matched via vector search).",
    )
    tmdb_id: Optional[int] = Field(
        default=None,
        description="Exact TMDB movie ID.  Takes precedence over title if both are given.",
    )

    # ------------------------------------------------------------------
    # WHY model_validator instead of field_validator:
    #   We need cross-field logic — "at least one of two fields" — which
    #   cannot be expressed on a single field.
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def at_least_one_provided(self):
        """Ensure the caller provides at least one identifier."""
        if not self.title and not self.tmdb_id:
            raise ValueError("Provide at least one of: title or tmdb_id")
        return self


# ---------------------------------------------------------------------------
# Explainability Sub-Schema
# ---------------------------------------------------------------------------

class GraphExplanation(BaseModel):
    """Structured explanation of the Knowledge Graph connection between
    the seed movie and a recommended candidate.

    WHY a dedicated schema:
        Separating graph reasoning from the overall explanation lets
        front-end consumers render a "knowledge graph" badge or tooltip
        independently of the free-text explanation string.
    """

    has_connection: bool = Field(
        description="True if any edge exists between seed and candidate in the graph.",
    )
    connection_detail: str = Field(
        description=(
            'Human-readable detail, e.g. "Shares director Christopher Nolan '
            'and actor Cillian Murphy".'
        ),
    )
    graph_boost: float = Field(
        description="Numeric boost applied from the graph (0.0–1.0).",
    )


# ---------------------------------------------------------------------------
# Recommendation Result
# ---------------------------------------------------------------------------

class RecommendationResult(BaseModel):
    """A single movie recommendation with full XAI provenance.

    Every numeric score and its qualitative explanation travel together
    so that consumers can choose between programmatic and human-readable
    representations.
    """

    tmdb_id: int
    title: str
    cosine_similarity: float = Field(
        description="Raw cosine similarity from vector search (0.0–1.0).",
    )
    graph_boost: float = Field(
        description="Additive boost from shared director / cast edges (0.0–1.0).",
    )
    final_score: float = Field(
        description="min(cosine_similarity + graph_boost, 1.0)",
    )
    explanation: str = Field(
        description="Full human-readable XAI string combining semantic and graph reasoning.",
    )
    graph_explanation: GraphExplanation


# ---------------------------------------------------------------------------
# Response Wrappers
# ---------------------------------------------------------------------------

class RecommendResponse(BaseModel):
    """Envelope for a successful /recommend response."""

    seed_title: str
    seed_tmdb_id: int
    results: List[RecommendationResult]


class ErrorResponse(BaseModel):
    """Standardised error body returned on 404 and other handled failures."""

    error: str
    detail: str
