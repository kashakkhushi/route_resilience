"""
Layer 6 - Visualization.

Matplotlib figures for the report and for the Streamlit dashboard. Kept free of
Streamlit imports so the same functions serve both the app and the batch report.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection


def _edge_segments(G: nx.Graph, simplify: bool = True):
    """Return (segments, kinds) in (x, y) order for LineCollection."""
    segs, kinds = [], []
    for u, v, d in G.edges(data=True):
        pts = d.get("pts")
        if not simplify and pts is not None and len(pts) > 1:
            arr = np.asarray(pts, float)[:, ::-1]  # (row,col) -> (x,y)
        else:
            pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
            arr = np.array([[pu[1], pu[0]], [pv[1], pv[0]]])
        segs.append(arr)
        kinds.append(d.get("kind", "observed"))
    return segs, kinds


def plot_network(G, background=None, ax=None, show_healed=True, title="Road network"):
    """Draw the graph, colouring healed bridges distinctly so a reviewer can see
    exactly what the algorithm invented versus what the model observed."""
    ax = ax or plt.subplots(figsize=(9, 9))[1]
    if background is not None:
        ax.imshow(background, cmap="gray" if background.ndim == 2 else None,
                  alpha=0.85 if background.ndim == 2 else 1.0)

    segs, kinds = _edge_segments(G, simplify=False)
    obs = [s for s, k in zip(segs, kinds) if k != "healed"]
    heal = [s for s, k in zip(segs, kinds) if k == "healed"]

    ax.add_collection(LineCollection(obs, colors="#1f77b4", linewidths=1.6, label="observed"))
    if show_healed and heal:
        ax.add_collection(LineCollection(heal, colors="#d62728", linewidths=2.2,
                                         linestyles="--", label="healed (MST bridge)"))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    return ax


def plot_criticality_heatmap(G, scores: dict, ax=None, top_n: int = 15,
                             gatekeepers=None, title="Criticality heatmap"):
    """Nodes sized and coloured by criticality; gatekeepers ringed."""
    ax = ax or plt.subplots(figsize=(9, 9))[1]
    segs, _ = _edge_segments(G, simplify=False)
    ax.add_collection(LineCollection(segs, colors="#cccccc", linewidths=1.0))

    nodes = list(G.nodes)
    xy = np.array([[G.nodes[n]["pos"][1], G.nodes[n]["pos"][0]] for n in nodes])
    vals = np.array([scores.get(n, 0.0) for n in nodes])

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=12 + 180 * vals,
                    cmap="inferno", vmin=0, vmax=1, zorder=3, edgecolors="none")
    if gatekeepers:
        gk = [i for i, n in enumerate(nodes) if n in set(gatekeepers)]
        if gk:
            ax.scatter(xy[gk, 0], xy[gk, 1], s=140, facecolors="none",
                       edgecolors="#00d0ff", linewidths=1.1, zorder=4,
                       label="gatekeeper")
            ax.legend(loc="upper right", fontsize=8)

    order = np.argsort(-vals)[:top_n]
    for i in order:
        ax.annotate(str(nodes[i]), (xy[i, 0], xy[i, 1]), fontsize=7,
                    color="white", zorder=5,
                    xytext=(4, 4), textcoords="offset points")

    plt.colorbar(sc, ax=ax, fraction=0.04, label="criticality")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_scenario(G, scenario, ax=None, title=None):
    """Intact network in grey, disrupted elements in red."""
    ax = ax or plt.subplots(figsize=(9, 9))[1]
    segs, _ = _edge_segments(G, simplify=False)
    ax.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=1.2))

    removed = set(scenario.removed_nodes)
    hit = [(u, v, d) for u, v, d in G.edges(data=True) if u in removed or v in removed]
    hit += [(u, v, G[u][v]) for u, v in scenario.removed_edges if G.has_edge(u, v)]
    if hit:
        rs = [np.asarray(d.get("pts", []), float)[:, ::-1] if len(d.get("pts", [])) > 1
              else np.array([[G.nodes[u]["pos"][1], G.nodes[u]["pos"][0]],
                             [G.nodes[v]["pos"][1], G.nodes[v]["pos"][0]]])
              for u, v, d in hit]
        ax.add_collection(LineCollection(rs, colors="#d62728", linewidths=2.4))
    if removed:
        xy = np.array([[G.nodes[n]["pos"][1], G.nodes[n]["pos"][0]]
                       for n in removed if n in G])
        if len(xy):
            ax.scatter(xy[:, 0], xy[:, 1], c="#d62728", s=30, marker="x", zorder=4)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title or f"Scenario: {scenario.name}")
    ax.axis("off")
    return ax


def plot_reroute(G, reroute: dict, ax=None):
    """Original route (blue) vs post-disruption alternate (orange)."""
    ax = ax or plt.subplots(figsize=(9, 9))[1]
    segs, _ = _edge_segments(G, simplify=False)
    ax.add_collection(LineCollection(segs, colors="#dddddd", linewidths=1.0))

    def draw(path, color, label, lw):
        if not path:
            return
        xy = np.array([[G.nodes[n]["pos"][1], G.nodes[n]["pos"][0]] for n in path])
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, label=label, zorder=3)

    draw(reroute["original_path"], "#1f77b4", "original", 2.5)
    draw(reroute["alternate_path"], "#ff7f0e", "alternate", 2.5)

    for key, color, marker in [("source", "green", "o"), ("target", "black", "s")]:
        n = reroute[key]
        if n in G:
            p = G.nodes[n]["pos"]
            ax.scatter([p[1]], [p[0]], c=color, s=70, marker=marker, zorder=5)

    dr = reroute.get("detour_ratio")
    ax.set_title(f"Rerouting - detour {dr}x" if dr else "Rerouting - destination unreachable")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    return ax


def plot_robustness(curves: dict, ax=None):
    """Random vs targeted degradation. The gap is the headline finding."""
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    styles = {"random": ("#2ca02c", "-o"), "targeted": ("#d62728", "-s"),
              "degree": ("#9467bd", "-^")}
    for name, c in curves.items():
        color, style = styles.get(name, ("#333333", "-o"))
        ax.plot(c["fraction_removed"], c["efficiency_retained"], style, color=color,
                markersize=4, label=f"{name} (AUC {c['auc_robustness']:.3f})")
    ax.set_xlabel("Fraction of intersections removed")
    ax.set_ylabel("Network efficiency retained")
    ax.set_title("Robustness under progressive failure")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)
    return ax


def plot_occlusion_demo(clean, occluded, mask, alpha, path=None):
    """Four-panel figure for the report: what the occluder does to the input
    while the label stays unchanged."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, (im, t, kw) in zip(axes, [
        (clean, "Clean scene", {}),
        (occluded, "Occluded input (fed to model)", {}),
        (mask, "Ground-truth road (UNCHANGED)", {"cmap": "gray"}),
        (alpha, "Occlusion alpha (aux-head target)", {"cmap": "magma"}),
    ]):
        ax.imshow(im, **kw)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
