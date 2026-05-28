"""
vector_engine.py — ChromaDB Semantic Vector Search Engine
==============================================================================

WHY ChromaDB OVER FAISS:
    ChromaDB provides built-in persistence, metadata filtering, and a
    Pythonic API. FAISS is faster for raw ANN but lacks these features.
    At ~1000 documents the speed difference is irrelevant.

WHY MANUAL EMBEDDINGS:
    Explicit control over the model lifecycle, batch pre-computation
    during setup, and portability if we swap vector stores later.

KNOWN LIMITATIONS:
    - all-MiniLM-L6-v2 is optimised for English; non-English synopses degrade.
    - Cosine similarity captures thematic similarity but misses factual
      links — that's why we layer the Knowledge Graph on top.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class VectorEngine:
    """Manages ChromaDB collection and sentence-transformer embeddings."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "movie_synopses"

    def __init__(self, persist_path: Optional[str] = None):
        """Load embedding model and connect to ChromaDB collection."""
        self.persist_path = persist_path or settings.chroma_persist_path
        logger.info("Loading sentence-transformer model '%s' …", self.MODEL_NAME)
        self.model = SentenceTransformer(self.MODEL_NAME)

        # PersistentClient survives restarts without re-indexing
        self.client = chromadb.PersistentClient(path=self.persist_path)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB '%s' ready (%d docs).", self.COLLECTION_NAME, self.collection.count()
        )

    def index_movie(self, tmdb_id: int, title: str, synopsis: str,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Embed synopsis and upsert into ChromaDB (idempotent)."""
        if not synopsis.strip():
            logger.warning("Skipping movie %d (%s) — empty synopsis.", tmdb_id, title)
            return

        embedding = self.model.encode(synopsis, show_progress_bar=False).tolist()
        meta: Dict[str, Any] = {"title": title, "tmdb_id": tmdb_id}
        if metadata:
            for key in ("director", "cast", "genres", "keywords"):
                val = metadata.get(key)
                if isinstance(val, list):
                    meta[key] = ", ".join(str(v) for v in val)
                elif val is not None:
                    meta[key] = str(val)

        self.collection.upsert(
            ids=[str(tmdb_id)], embeddings=[embedding],
            documents=[synopsis], metadatas=[meta],
        )

    def search(self, query_text: str, top_k: int = 20) -> List[Dict]:
        """Embed query and return top-k similar movies with cosine similarity."""
        if not query_text.strip():
            return []

        query_embedding = self.model.encode(query_text, show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns distances; cosine_sim = 1 - distance
        output: List[Dict] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for doc_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
            cosine_sim = max(1.0 - dist, 0.0)
            output.append({
                "tmdb_id": int(doc_id),
                "title": meta.get("title", ""),
                "cosine_similarity": round(cosine_sim, 4),
                "director": meta.get("director", ""),
                "cast": meta.get("cast", ""),
                "genres": meta.get("genres", ""),
                "synopsis": doc,
            })
        return output

    def get_by_tmdb_id(self, tmdb_id: int) -> Optional[Dict]:
        """Retrieve a single movie from ChromaDB by TMDB ID."""
        try:
            result = self.collection.get(
                ids=[str(tmdb_id)], include=["documents", "metadatas"],
            )
            if not result["ids"]:
                return None
            meta = result["metadatas"][0]
            return {
                "tmdb_id": int(result["ids"][0]),
                "title": meta.get("title", ""),
                "synopsis": result["documents"][0] if result["documents"] else "",
                "director": meta.get("director", ""),
                "cast": meta.get("cast", ""),
                "genres": meta.get("genres", ""),
            }
        except Exception:
            return None

    def count(self) -> int:
        """Return total indexed documents."""
        return self.collection.count()
