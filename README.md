#  Hybrid Movie Recommender System

A modular, resource-efficient **Hybrid Movie Recommender** that combines **Semantic Vector Search** (for conceptual similarity) with a **Knowledge Graph** (for factual links like shared Directors and Actors), wrapped in a structured **Explainability (XAI) layer**. 

Now includes a **sleek, interactive web UI** with glassmorphism design, dynamic autocomplete, and live movie posters fetched from TMDB!

Built as a technical portfolio piece demonstrating ML engineering best practices.

---

## Architecture Overview

```
                    ┌───────────────┐
                    │   FastAPI     │
                    │   /recommend  │
                    └──────┬────────┘
                           │
                    ┌──────▼────────┐
                    │   Hybrid      │
                    │   Re-Ranker   │
                    │   + XAI       │
                    └──┬────────┬───┘
                       │        │
              ┌────────▼──┐  ┌──▼──────────┐
              │  ChromaDB  │  │  NetworkX   │
              │  Semantic  │  │  Knowledge  │
              │  Search    │  │  Graph      │
              └────────────┘  └─────────────┘
                       │        │
              ┌────────▼────────▼───┐
              │   SQLite Cache      │
              │   (TMDB Data)       │
              └─────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

### 2. Clone and Install Dependencies

```bash
git clone https://github.com/VinukaKulathunge/WhatToWatchNext.git
cd WhatToWatchNext
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your TMDB API key
```

### 4. Run Setup (One-Time)

```bash
python setup.py
```

This fetches ~1,000 movies from TMDB, embeds their synopses, and builds the knowledge graph.

** Estimated Runtime:** 10-20 minutes for 1,000 movies (dominated by TMDB API rate limits at ~40 requests per 10 seconds). The setup is **idempotent** — if interrupted, re-run to continue from where it left off.

### 5. Start the API

```bash
uvicorn main:app --reload
```

The sleek Web UI will be available directly at `http://localhost:8000`. You can also find the interactive API docs at `http://localhost:8000/docs`.

---

## API Usage

### POST `/recommend` — Get Recommendations

```bash
# By title (fuzzy-matched via vector search)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception"}'

# By TMDB ID (exact match)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"tmdb_id": 27205}'
```

**Example Response:**

```json
{
  "seed_title": "Inception",
  "seed_tmdb_id": 27205,
  "results": [
    {
      "tmdb_id": 157336,
      "title": "Interstellar",
      "cosine_similarity": 0.7823,
      "graph_boost": 0.4,
      "final_score": 1.0,
      "explanation": "Recommended because the synopsis is 78% semantically similar to Inception. Also shares director Christopher Nolan with the seed movie.",
      "graph_explanation": {
        "has_connection": true,
        "connection_detail": "Shares director Christopher Nolan",
        "graph_boost": 0.4
      }
    }
  ]
}
```

### GET `/health` — Health Check

```bash
curl http://localhost:8000/health
```

### GET `/movie/{tmdb_id}` — Raw Movie Data

```bash
curl http://localhost:8000/movie/27205
```

---

## Engineering Trade-offs

### Why Hybrid Beats Pure Semantic Search

Pure semantic (vector) search finds movies with similar *themes, tone, and narrative structure* — e.g., querying "Inception" surfaces other mind-bending sci-fi films. But it **misses factual connections**: two Christopher Nolan films with very different plots (say, *Dunkirk* and *Tenet*) may not rank highly by synopsis similarity alone. The knowledge graph captures these deterministic relationships (same director, shared actors) and boosts candidates accordingly. This also helps with **cold start**: a new movie with a thin synopsis but a well-known director still gets meaningful connections.

### Why NetworkX Over a Hosted Graph DB

At ~1,000 nodes and ~5,000–15,000 edges, the entire graph fits in <10 MB of RAM. NetworkX provides:
- **Zero infrastructure** — no Neo4j server to install, configure, or maintain.
- **Instant startup** — unpickling takes <100 ms.
- **Rich algorithms** — shortest path, centrality, community detection are one import away if needed later.

For a portfolio project, the simplicity / capability ratio is unbeatable. A hosted graph DB would be justified at >100K nodes with concurrent writes.

### Why all-MiniLM-L6-v2 Over Larger Models

| Model | Parameters | Embedding Speed (CPU) | STS Benchmark |
|-------|-----------|----------------------|---------------|
| all-MiniLM-L6-v2 | 22M | ~2,500 sent/sec | 0.789 |
| all-mpnet-base-v2 | 109M | ~500 sent/sec | 0.838 |

For synopsis-length text (1–3 sentences), the quality gap is marginal while the speed advantage is 5×. The smaller model also loads in <1 GB RAM, making it deployable on modest hardware (e.g., a $5/month VPS).

### Why ChromaDB Over FAISS

| Feature | ChromaDB | FAISS |
|---------|----------|-------|
| Persistence | Built-in | Manual (serialize/load) |
| Metadata filtering | Native | None (external join required) |
| API simplicity | High (Pythonic) | Medium (C++ bindings) |
| Raw ANN speed | Good | Excellent |

At portfolio scale (~1,000 vectors), FAISS's speed advantage is imperceptible. ChromaDB's persistence and metadata filtering save significant boilerplate.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Graph sparsity for niche films** | Indie films with unique cast/directors have few or no graph edges → no boost. | The system falls back gracefully to pure semantic ranking. |
| **TMDB data quality** | Some movies have empty overviews, wrong credits, or missing keywords. | Empty synopses are skipped during embedding; movies can be re-fetched. |
| **Synopsis language bias** | all-MiniLM-L6-v2 is trained on English text; non-English synopses produce lower-quality embeddings. | TMDB is queried with `language=en-US` to prefer English overviews. |
| **Additive scoring formula** | `cosine_sim + graph_boost` treats both signals as equally trustworthy. A learned weighting would be more nuanced. | The current formula is simple, interpretable, and effective for a portfolio demo. |
| **Single-process API** | No horizontal scaling without shared state (e.g., Redis, PostgreSQL). | Sufficient for demo / portfolio use. |

---

## Project Structure

```
WhatToWatchNext/
├── setup.py               # One-time data fetch, embed, graph build
├── data_manager.py        # TMDB fetching, SQLite caching
├── vector_engine.py       # ChromaDB indexing and semantic search
├── graph_engine.py        # NetworkX graph build, edge weighting, pickle I/O
├── recommender.py         # Hybrid re-ranker, scoring formula, XAI logic
├── main.py                # FastAPI app, /recommend endpoint, startup events
├── models.py              # All Pydantic request/response schemas
├── config.py              # Loads .env, exposes typed settings object
├── static/                # Frontend UI assets (HTML, CSS, JS)
├── data/                  # SQLite database (auto-created)
├── graph/                 # Pickled NetworkX graph (auto-created)
├── chroma_db/             # ChromaDB persistent storage (auto-created)
├── .env.example           # Environment variable template
├── requirements.txt       # Pinned Python dependencies
└── README.md              # This file
```

---

## License

This project is intended as a portfolio demonstration. Feel free to use, modify, and learn from it.
