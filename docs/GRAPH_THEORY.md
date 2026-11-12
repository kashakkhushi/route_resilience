# Graph Theory Reference

## 1. Construction

Skeletonisation reduces the mask to one-pixel centrelines. The skeleton becomes
`G = (V, E)` where nodes are intersections and endpoints, edges are segments.

Three details decide whether the graph is usable:

- **Junction clustering.** A 4-way intersection is a blob of 3–8 pixels with
  degree ≥ 3. Treating each as a node produces a hairball of zero-length edges
  and wrecks centrality. Connected components of junction pixels collapse to one
  node.
- **Spur pruning.** Skeletonisation throws off short dead-end barbs that are not
  roads. Anything shorter than `min_spur_m` ending in a degree-1 node is removed,
  iteratively.
- **Degree-2 chaining.** Interior pixels must not become nodes, or betweenness
  becomes intractable. Geometry is preserved in the edge's `pts`.

Edge attributes: `length_px`, `length_m`, `kind ∈ {observed, healed}`, `pts`.

## 2. Topological healing

**Disjoint Set (union–find)** with path compression and union by rank tracks which
fragments are already merged in near-O(1).

**Kruskal's MST** over cost-sorted candidate bridges connects `k` components using
exactly `k − 1` bridges — provably minimal. The union–find check rejects any
bridge between fragments already linked by a cheaper chain.

```
cost = d · (1 + w_angle · ang_penalty) · (1 + w_evidence · (1 − evidence))

d            Euclidean gap between endpoints
ang_penalty  0 when the bridge continues both roads' direction, 1 when it doubles back
evidence     mean (road probability + occlusion probability) along the bridge
```

Bridges above `max_cost` are refused outright.

### The loop-restoration pass

A spanning tree **cannot** repair a gap whose two sides remain connected the long
way round — a break in a ring road, or in any grid block. Union–find correctly
says "already connected", and the loop is destroyed. Connectivity then looks
perfect while redundancy collapses.

The second pass re-examines intra-component endpoint pairs and bridges those
where the graph distance exceeds `detour_threshold ×` the direct gap.

Measured against known ground truth on the synthetic benchmark:

| | ground truth | MST only | MST + loop pass |
|---|---|---|---|
| α index | 0.132 | 0.018 | 0.075 |
| gatekeepers | 57 | 117 | 52 |
| bridge edges | 70 | 183 | 65 |

## 3. Centrality

**Betweenness** (Brandes, weighted by `length_m`): the share of shortest paths
through a node. High betweenness = load-bearing corridor.

**Articulation points** (gatekeepers): removing one *increases the component
count*. Binary, not gradual.

**Bridge edges**: the edge analogue — typically literal bridges, flyovers, level
crossings.

Busy ≠ critical. High-betweenness nodes usually have alternatives, which is often
why they carry traffic. Gatekeepers have none.

**α index (meshedness)** = `(m − n + c) / (2n − 5c)`, clipped to [0, 1].
0 = tree, every edge a single point of failure. Grid cities land near 0.2–0.4.

## 4. Failure simulation

`G' = G − v` or `G − e`, evaluated on a **fixed** OD sample drawn once from the
intact network. Resampling per scenario silently changes the question.

**Sequential ablation with re-ranking.** After the top node is removed the network
reorganises, so the next most critical node is usually *not* the original number
two. Batch removal of an initially ranked list understates damage; sequential
removal is the honest upper bound.

## 5. Resilience Index (primary metric: NER)

**Network Efficiency Resilience.**

```
NER  =  E(G')  /  E(G)

E(H) =  (1 / |P|)  *  SUM over (s,t) in P  of  1 / d_H(s,t),     with 1/inf := 0

G    intact network
G'   network after the failure scenario
P    fixed origin-destination sample, drawn ONCE from G and reused for both terms
d_H  shortest-path length in H, weighted by length_m (or travel_time_s if present)
```

**Range: `0 <= NER <= 1`, by construction.**

| Value | Meaning |
|---|---|
| `1.00` | no sampled route changed; nothing was lost |
| `0.63` | 63% of baseline routing efficiency retained, 37% lost |
| `0.00` | every sampled trip is severed |

### Why it is bounded, and why that is a theorem rather than a clamp

Removing nodes or edges can never create a new path, so for every pair
`d_G'(s,t) >= d_G(s,t)`. Taking reciprocals termwise gives
`1/d_G' <= 1/d_G`, hence `E(G') <= E(G)` and `NER <= 1`. Every term is
non-negative, so `NER >= 0`. No clamping is applied; the only guard in the code
is against floating-point round-off at the 1e-12 level.

Three further properties follow, and each has a regression test:

- **Identity** — an empty scenario scores exactly `1.0`.
- **Monotonicity** — for nested removals `G'' ⊂ G' ⊂ G`, `NER(G'') <= NER(G')`.
  More damage can never raise the score.
- **Totality** — a severed pair contributes `0` rather than infinity, so
  disconnection *lowers* the score instead of making it undefined. Detour and
  disconnection are captured by one number.

### Two secondary readings that are NOT the index

Both are reported for completeness and are deliberately kept out of the primary
field.

**Literal ASPL ratio** (`resilience_index_literal_aspl`) — the project brief's
definition, `ASPL_baseline / ASPL_perturbed`. Mean shortest path length becomes
infinite the moment one OD pair disconnects, so this reads exactly `0.00` for
almost every realistic scenario and cannot rank them. Flagged with
`literal_aspl_is_degenerate`.

**Connected-only ASPL ratio** (`diagnostic_aspl_ratio_connected_only`) —
averages over surviving pairs only. When most trips are severed the survivors
are disproportionately *short* ones, so mean path length falls and the ratio
climbs **above 1**, which would read as a collapsed network improving. Measured
example: a chain network shattered into fragments, 92% of trips impossible,
yields `5.33`. It carries `diagnostic_is_resilience_score: false` and
`diagnostic_survivorship_biased: true`. It is kept because it is a useful
diagnostic of *which* trips survived, and removed from the primary slot because
it is unbounded and non-monotone.


## 6. Evaluation

| Metric | Measures |
|---|---|
| IoU / Dice | pixel accuracy — necessary, not sufficient |
| clDice | centreline (topology) agreement |
| Relaxed IoU (3–5 px buffer) | tolerance to sub-pixel misalignment |
| Occlusion-Recall | recall split visible vs occluded — the project's core claim |
| Connectivity Ratio | largest-component growth after healing |
| APLS | symmetric route-similarity vs OSM |

APLS is symmetric (harmonic mean of both directions) because GT→pred alone
rewards omitting roads, and pred→GT alone rewards hallucinating them.
