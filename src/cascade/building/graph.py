"""Water-path graph and electrical tree over a Building (building_spec section 4).

Pure Python: dict adjacency and heapq, no networkx. Water edges: drains_to and above are directed downstream,
adjacent goes both ways. Electrical edges: feeds, parent to child. Weights default to rules EDGE_WEIGHT.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .model import WATER_EDGE_KINDS, Building
from .rules import t

Adjacency = Dict[str, List[Tuple[str, float, str]]]


@dataclass(frozen=True)
class Reach:
    score: float  # source strength times the product of edge weights on the best path
    hops: int
    path: Tuple[str, ...]  # from the source to the reached node


def _weight(kind: str, weight: Optional[float]) -> float:
    return float(weight) if weight is not None else float(t("EDGE_WEIGHT")[kind])


def water_adjacency(b: Building) -> Adjacency:
    """Downstream adjacency: node -> [(next node, weight, edge kind)]. adjacent edges appear in both directions."""

    def build() -> Adjacency:
        adj: Adjacency = {}
        for e in b.edges:
            if e.kind not in WATER_EDGE_KINDS:
                continue
            w = _weight(e.kind, e.weight)
            adj.setdefault(e.src, []).append((e.dst, w, e.kind))
            if e.kind == "adjacent":
                adj.setdefault(e.dst, []).append((e.src, w, e.kind))
        return adj

    return b.memo("graph.water_adjacency", build)


def reverse(adj: Adjacency) -> Adjacency:
    out: Adjacency = {}
    for src, nbrs in adj.items():
        for dst, w, kind in nbrs:
            out.setdefault(dst, []).append((src, w, kind))
    return out


def reach(adj: Adjacency, sources: Dict[str, float], max_hops: int = 8, min_score: Optional[float] = None) -> Dict[str, Reach]:
    """Best (max-product) path score from any source to every reachable node: Dijkstra on -log weight, with a hop
    limit. Sources themselves are included with hops 0. Nodes below min_score are dropped (default: none)."""
    best: Dict[str, Reach] = {}
    heap: List[Tuple[float, int, str, Tuple[str, ...], float]] = []
    for node, strength in sorted(sources.items()):
        if strength > 0:
            heapq.heappush(heap, (-math.log(strength), 0, node, (node,), strength))
    while heap:
        _, hops, node, path, score = heapq.heappop(heap)
        if node in best:
            continue
        if min_score is not None and score < min_score:
            continue
        best[node] = Reach(score=score, hops=hops, path=path)
        if hops >= max_hops:
            continue
        for nxt, w, _kind in adj.get(node, []):
            if nxt in best or w <= 0:
                continue
            s = score * w
            if min_score is not None and s < min_score:
                continue
            heapq.heappush(heap, (-math.log(s), hops + 1, nxt, path + (nxt,), s))
    return best


def downstream(b: Building, node: str, max_hops: int = 8) -> Dict[str, Reach]:
    """Where water starting at `node` can go (node itself included, hops 0)."""
    return reach(water_adjacency(b), {node: 1.0}, max_hops=max_hops)


def upstream(b: Building, node: str, max_hops: int = 8) -> Dict[str, Reach]:
    """Where water reaching `node` can come from (node itself included). Reach.path runs from `node` upstream."""
    rev = b.memo("graph.water_reverse", lambda: reverse(water_adjacency(b)))
    return reach(rev, {node: 1.0}, max_hops=max_hops)


def water_sources(b: Building) -> List[str]:
    """Nodes where water enters the building: facade drops and roof zones, roof drains and risers."""
    zs = [z.zone_id for z in b.zones if z.kind in ("facade_drop", "roof")]
    es = [e.element_id for e in b.elements if e.kind in ("roof_drain", "riser")]
    return zs + es


def electrical_children(b: Building) -> Dict[str, List[str]]:
    def build() -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for e in b.edges:
            if e.kind == "feeds":
                out.setdefault(e.src, []).append(e.dst)
        return out

    return b.memo("graph.electrical_children", build)


def electrical_parent(b: Building) -> Dict[str, str]:
    return b.memo("graph.electrical_parent", lambda: {e.dst: e.src for e in b.edges if e.kind == "feeds"})


def subtree(b: Building, node: str) -> List[str]:
    """`node` plus every descendant it feeds, depth first."""
    kids = electrical_children(b)
    out: List[str] = []
    stack = [node]
    seen = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        stack.extend(reversed(kids.get(n, [])))
    return out


def path_to_root(b: Building, node: str) -> List[str]:
    """[node, parent, grandparent, ..., root] along feeds edges."""
    parent = electrical_parent(b)
    out = [node]
    seen = {node}
    while out[-1] in parent and parent[out[-1]] not in seen:
        out.append(parent[out[-1]])
        seen.add(out[-1])
    return out


def zone_of(b: Building, node: str) -> Optional[str]:
    """A zone id itself, an element's zone, or a sensor's zone."""
    if b.zone(node) is not None:
        return node
    e = b.element(node)
    if e is not None:
        return e.zone_id
    s = b.sensor(node)
    return s.zone_id if s is not None else None
