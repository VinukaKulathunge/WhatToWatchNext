"""
graph_engine.py — NetworkX Knowledge Graph for Factual Movie Links
==============================================================================

WHY NetworkX OVER A HOSTED GRAPH DB (e.g. Neo4j):
    At ~1 000 nodes the entire graph fits in < 10 MB of RAM.  NetworkX
    gives us zero infrastructure overhead, instant startup (unpickle),
    and a rich algorithmic library — all without running a server.

WHAT PROBLEM IT SOLVES:
    Pure semantic search misses *factual* connections: "these two films
    share a director" or "the same lead actor stars in both".  The graph
    captures these deterministic relationships as weighted edges so the
    hybrid re-ranker can boost genuinely related movies.

EDGE WEIGHTING RATIONALE:
    Director edges (0.4) are weighted higher than individual actor edges
    (0.2 each) because a shared director is a stronger signal of stylistic
    similarity than a shared actor.  Combined max = 1.0.

KNOWN LIMITATIONS:
    - Graph becomes sparse for niche / indie films with unique cast.
    - Pickle format is Python-version–sensitive; upgrading Python may
      require rebuilding the graph.
"""

import pickle
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# Edge weight constants — tuned for director > actor signal strength
DIRECTOR_WEIGHT = 0.4
ACTOR_WEIGHT = 0.2   # per shared actor
MAX_ACTOR_WEIGHT = 0.6
MAX_COMBINED_WEIGHT = 1.0


def build_graph(movies: List[Dict]) -> nx.Graph:
    """Construct an undirected weighted graph from a list of movie dicts.

    Each movie dict must contain: tmdb_id, title, director, cast (list).

    Algorithm:
        1. Add every movie as a node with title + director attributes.
        2. Build reverse indices: director→movies, actor→movies.
        3. For each pair of movies that share director or cast, create
           or update an edge with the combined weight.

    WHY reverse indices:
        Naively comparing every pair is O(n²).  Reverse indices let us
        iterate only over movies that actually share a person — typically
        a much smaller set.
    """
    G = nx.Graph()

    # Pass 1: add nodes
    for m in movies:
        G.add_node(
            m["tmdb_id"],
            title=m.get("title", ""),
            director=m.get("director", ""),
        )

    # Pass 2: build reverse indices
    director_index: Dict[str, List[int]] = defaultdict(list)
    actor_index: Dict[str, List[int]] = defaultdict(list)

    for m in movies:
        d = m.get("director", "").strip()
        if d:
            director_index[d].append(m["tmdb_id"])
        for actor in m.get("cast", [])[:3]:
            a = actor.strip() if isinstance(actor, str) else str(actor)
            if a:
                actor_index[a].append(m["tmdb_id"])

    # Pass 3: create / update edges from director links
    for director, ids in director_index.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                _add_or_update_edge(G, a, b, "director", director, DIRECTOR_WEIGHT)

    # Pass 4: create / update edges from actor links
    for actor, ids in actor_index.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                _add_or_update_edge(G, a, b, "actor", actor, ACTOR_WEIGHT)

    logger.info(
        "Graph built: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges()
    )
    return G


def _add_or_update_edge(
    G: nx.Graph, a: int, b: int, link_type: str, name: str, weight: float
) -> None:
    """Add a weighted edge or update an existing one.

    Stores shared_directors and shared_actors lists on each edge for
    explainability, and accumulates weight up to MAX_COMBINED_WEIGHT.
    """
    if G.has_edge(a, b):
        data = G[a][b]
    else:
        data = {"weight": 0.0, "shared_directors": [], "shared_actors": []}
        G.add_edge(a, b, **data)
        data = G[a][b]

    if link_type == "director" and name not in data["shared_directors"]:
        data["shared_directors"].append(name)
        data["weight"] = min(data["weight"] + weight, MAX_COMBINED_WEIGHT)
    elif link_type == "actor" and name not in data["shared_actors"]:
        # Cap actor contribution at MAX_ACTOR_WEIGHT
        current_actor_weight = len(data["shared_actors"]) * ACTOR_WEIGHT
        if current_actor_weight < MAX_ACTOR_WEIGHT:
            data["shared_actors"].append(name)
            data["weight"] = min(data["weight"] + weight, MAX_COMBINED_WEIGHT)


def add_movie_to_graph(G: nx.Graph, movie: Dict, all_movies: List[Dict]) -> None:
    """Add a single movie node and compute its edges against existing nodes.

    Used by the cold-start handler to extend the graph at runtime.
    """
    tid = movie["tmdb_id"]
    G.add_node(tid, title=movie.get("title", ""), director=movie.get("director", ""))

    m_dir = movie.get("director", "").strip()
    m_cast = set(
        a.strip() for a in movie.get("cast", [])[:3] if isinstance(a, str) and a.strip()
    )

    for other in all_movies:
        oid = other["tmdb_id"]
        if oid == tid:
            continue

        # Director edge
        o_dir = other.get("director", "").strip()
        if m_dir and o_dir and m_dir == o_dir:
            _add_or_update_edge(G, tid, oid, "director", m_dir, DIRECTOR_WEIGHT)

        # Actor edges
        o_cast = set(
            a.strip() for a in other.get("cast", [])[:3]
            if isinstance(a, str) and a.strip()
        )
        shared = m_cast & o_cast
        for actor in shared:
            _add_or_update_edge(G, tid, oid, "actor", actor, ACTOR_WEIGHT)


# ---------------------------------------------------------------------------
# Persistence (pickle I/O)
# ---------------------------------------------------------------------------

def save_graph(graph: nx.Graph, path: Optional[str] = None) -> None:
    """Pickle the graph to disk."""
    from config import settings
    path = path or settings.graph_pickle_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Graph saved to %s", path)


def load_graph(path: Optional[str] = None) -> nx.Graph:
    """Load a pickled graph from disk."""
    from config import settings
    path = path or settings.graph_pickle_path
    with open(path, "rb") as f:
        graph = pickle.load(f)
    logger.info(
        "Graph loaded: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges()
    )
    return graph


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_graph_boost(graph: nx.Graph, seed_id: int, candidate_id: int) -> float:
    """Return the edge weight between two nodes, or 0.0 if no edge."""
    if graph.has_edge(seed_id, candidate_id):
        return graph[seed_id][candidate_id].get("weight", 0.0)
    return 0.0


def get_graph_explanation(graph: nx.Graph, seed_id: int, candidate_id: int) -> str:
    """Return a human-readable string explaining the graph connection.

    Examples:
        "Shares director Christopher Nolan and actor Cillian Murphy"
        "Shares actors Tom Hanks, Leonardo DiCaprio"
        "" (empty if no connection)
    """
    if not graph.has_edge(seed_id, candidate_id):
        return ""

    data = graph[seed_id][candidate_id]
    parts = []

    directors = data.get("shared_directors", [])
    actors = data.get("shared_actors", [])

    if directors:
        parts.append(f"director {', '.join(directors)}")
    if actors:
        label = "actor" if len(actors) == 1 else "actors"
        parts.append(f"{label} {', '.join(actors)}")

    if parts:
        return "Shares " + " and ".join(parts)
    return ""
