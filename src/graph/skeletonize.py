"""
Layer 4a - Skeletonization and graph extraction.

mask (H,W) -> 1-px skeleton -> networkx.Graph where
    nodes = intersections and endpoints, with 'pos' = (row, col)
    edges = road segments, with 'pts' (the traced pixel chain), 'length_px', 'geom_length'

The non-obvious parts, in order of how much trouble they cause if you skip them:

1. Junction clustering. A 4-way intersection in a skeleton is rarely one pixel -
   it is a small blob of 3-8 pixels each having degree >= 3. Treating each as a
   separate node produces a hairball of zero-length edges and wrecks centrality.
   We label connected components of junction pixels and collapse each to one node.

2. Spur pruning. Skeletonization of a blobby mask throws off short dead-end
   barbs that are not real roads. Anything shorter than `min_spur_len` and
   ending in a degree-1 node is removed, iteratively.

3. Degree-2 chaining. Interior pixels of a segment must not become nodes, or the
   graph has millions of nodes and betweenness centrality becomes intractable.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

# 8-connectivity kernel used to count skeleton neighbours per pixel.
_NEIGH_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def clean_mask(mask: np.ndarray, min_object_px: int = 64, max_hole_px: int = 64) -> np.ndarray:
    """Remove speckle and fill pinholes before skeletonizing.

    Order matters: filling holes first prevents the skeleton from developing
    spurious loops around small false-negative gaps inside wide roads."""
    m = mask.astype(bool)
    # skimage >= 0.26 renamed area_threshold/min_size to max_size; call
    # positionally so the same code works on both old and new releases.
    if max_hole_px > 0:
        m = remove_small_holes(m, max_hole_px)
    if min_object_px > 0:
        m = remove_small_objects(m, min_object_px)
    return m


def mask_to_skeleton(mask: np.ndarray, **clean_kw) -> np.ndarray:
    return skeletonize(clean_mask(mask, **clean_kw))


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    return ndimage.convolve(skel.astype(np.uint8), _NEIGH_KERNEL, mode="constant") * skel


def _node_pixels(skel: np.ndarray):
    """Return (endpoint_mask, junction_mask) using 8-neighbour degree."""
    deg = _neighbour_count(skel)
    return (deg == 1) & skel, (deg >= 3) & skel


def skeleton_to_graph(skel: np.ndarray, pixel_size_m: float = 1.0) -> nx.Graph:
    """Trace a 1-pixel skeleton into a node/edge graph."""
    skel = skel.astype(bool)
    endpoints, junctions = _node_pixels(skel)

    # --- 1. collapse junction blobs into single nodes ---------------------
    jlabels, n_j = ndimage.label(junctions, structure=np.ones((3, 3)))
    node_id_at = np.zeros(skel.shape, dtype=np.int32)  # 0 = not a node
    G = nx.Graph()
    next_id = 1

    if n_j:
        centroids = ndimage.center_of_mass(junctions, jlabels, range(1, n_j + 1))
        for lab in range(1, n_j + 1):
            nid = next_id
            next_id += 1
            r, c = centroids[lab - 1]
            G.add_node(nid, pos=(float(r), float(c)), kind="junction")
            node_id_at[jlabels == lab] = nid

    for r, c in zip(*np.nonzero(endpoints)):
        if node_id_at[r, c]:  # already part of a junction blob
            continue
        nid = next_id
        next_id += 1
        G.add_node(nid, pos=(float(r), float(c)), kind="endpoint")
        node_id_at[r, c] = nid

    # --- 2. walk every segment between node pixels ------------------------
    visited_edge = set()  # frozenset of the first step, prevents double tracing

    def neighbours(r, c):
        for dr, dc in _OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                yield rr, cc

    node_px = np.nonzero(node_id_at)
    for r, c in zip(*node_px):
        r, c = int(r), int(c)
        start_id = int(node_id_at[r, c])
        for nr, nc in neighbours(r, c):
            key = frozenset({(r, c), (nr, nc)})
            if key in visited_edge:
                continue
            visited_edge.add(key)

            # If the neighbour belongs to the same collapsed junction, skip:
            # it is an interior pixel of the blob, not a real segment.
            if node_id_at[nr, nc] == start_id:
                continue

            pts = [(r, c)]
            pr, pc = r, c
            cr, cc_ = nr, nc
            end_id = 0
            while True:
                pts.append((cr, cc_))
                if node_id_at[cr, cc_]:
                    end_id = int(node_id_at[cr, cc_])
                    break
                nxt = [p for p in neighbours(cr, cc_) if p != (pr, pc)]
                # Prefer 4-connected steps; diagonal shortcuts around a corner
                # otherwise cause the walk to skip a pixel and mis-measure length.
                nxt.sort(key=lambda p: abs(p[0] - cr) + abs(p[1] - cc_))
                if not nxt:
                    break  # dangling tail (shouldn't happen after clean_mask)
                visited_edge.add(frozenset({(cr, cc_), nxt[0]}))
                pr, pc = cr, cc_
                cr, cc_ = nxt[0]

            if end_id and end_id != start_id:
                length = _polyline_length(pts)
                if G.has_edge(start_id, end_id) and G[start_id][end_id]["length_px"] <= length:
                    continue  # keep the shorter of parallel segments
                G.add_edge(
                    start_id, end_id,
                    pts=pts,
                    length_px=length,
                    length_m=length * pixel_size_m,
                    kind="observed",
                )

    G.graph["pixel_size_m"] = pixel_size_m
    return G


def _polyline_length(pts) -> float:
    """Euclidean chain length; diagonal steps count sqrt(2), not 1."""
    a = np.asarray(pts, dtype=np.float64)
    if len(a) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


def prune_spurs(G: nx.Graph, min_spur_len: float = 15.0, iterations: int = 3) -> nx.Graph:
    """Iteratively drop short dead-end branches (skeletonization artefacts)."""
    G = G.copy()
    for _ in range(iterations):
        doomed = [
            n for n in G.nodes
            if G.degree(n) == 1
            and next(iter(G.edges(n, data=True)))[2]["length_px"] < min_spur_len
        ]
        if not doomed:
            break
        G.remove_nodes_from(doomed)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return G


def simplify_degree2(G: nx.Graph) -> nx.Graph:
    """Merge chains of degree-2 nodes into single edges.

    Keeps geometry (concatenated 'pts') so the map still draws correctly while
    the graph gets small enough for exact betweenness centrality."""
    G = G.copy()
    changed = True
    while changed:
        changed = False
        for n in [x for x in G.nodes if G.degree(x) == 2]:
            (a, da), (b, db) = [(nbr, d) for nbr, d in G[n].items()]
            if a == b:
                continue  # self-loop through n: leave it alone
            if G.has_edge(a, b):
                continue  # merging would create a multi-edge
            pts = _join_pts(da["pts"], db["pts"], G.nodes[n]["pos"])
            G.add_edge(a, b, pts=pts,
                       length_px=da["length_px"] + db["length_px"],
                       length_m=da["length_m"] + db["length_m"],
                       kind="observed" if da["kind"] == db["kind"] == "observed" else "mixed")
            G.remove_node(n)
            changed = True
    return G


def _join_pts(p1, p2, shared_pos):
    """Concatenate two pixel chains that meet at a shared node."""
    p1, p2 = list(p1), list(p2)
    sr, sc = shared_pos
    if np.hypot(p1[0][0] - sr, p1[0][1] - sc) < np.hypot(p1[-1][0] - sr, p1[-1][1] - sc):
        p1 = p1[::-1]
    if np.hypot(p2[-1][0] - sr, p2[-1][1] - sc) < np.hypot(p2[0][0] - sr, p2[0][1] - sc):
        p2 = p2[::-1]
    return p1 + p2[1:]


def build_raw_graph(mask: np.ndarray, pixel_size_m: float = 1.0,
                    min_spur_len: float = 15.0) -> nx.Graph:
    """mask -> pruned, simplified graph. Still fragmented; healing comes next."""
    skel = mask_to_skeleton(mask)
    G = skeleton_to_graph(skel, pixel_size_m)
    G = prune_spurs(G, min_spur_len)
    return simplify_degree2(G)
