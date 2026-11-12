"""
OSM ground-truth automation - the "zero-manual-effort" annotation pipeline.

Given a GeoTIFF, this produces a pixel-aligned binary road mask and a reference
graph, with no manual labelling. That is what makes training on Sentinel-2 /
LISS-IV / Cartosat-3 feasible inside a 30-hour window.

Pipeline:
    GeoTIFF -> bounds -> reproject to WGS84 -> OSMnx query -> reproject to raster
    CRS -> buffer by road-class width -> rasterize onto the exact raster grid

Three details that decide whether the labels are usable:

1. **Class-dependent buffer width.** OSM gives centrelines, not footprints. A
   motorway rendered at the same 3 m width as a residential lane produces labels
   that disagree with the imagery everywhere. Widths below are per-direction
   half-widths in metres, tuned to typical Indian urban carriageways.

2. **Resolution guard.** At Sentinel-2's 10 m GSD a 3.5 m residential road is a
   third of a pixel and simply is not observable. Training on labels the sensor
   cannot resolve teaches the model to hallucinate. `min_resolvable_width_m`
   drops those classes and reports what it dropped.

3. **Temporal mismatch.** OSM is current; the image has an acquisition date.
   New roads appear as false negatives, demolished ones as false positives.
   Always record the image date, and treat a sudden precision drop in one tile
   as a data problem before assuming it is a model problem.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# Half-width in metres per OSM highway class (so full carriageway = 2x).
ROAD_HALF_WIDTH_M = {
    "motorway": 9.0, "motorway_link": 5.0,
    "trunk": 7.5, "trunk_link": 4.5,
    "primary": 6.0, "primary_link": 4.0,
    "secondary": 5.0, "secondary_link": 3.5,
    "tertiary": 4.0, "tertiary_link": 3.0,
    "residential": 3.5, "unclassified": 3.0,
    "service": 2.5, "living_street": 3.0,
    "track": 2.0, "road": 3.0,
}
DEFAULT_HALF_WIDTH_M = 3.0

SENSOR_GSD_M = {"sentinel2": 10.0, "liss4": 5.8, "cartosat3": 0.25}


def raster_bounds_wgs84(raster_path: str | Path) -> tuple:
    """Return (west, south, east, north) in EPSG:4326 for an OSM query."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster_path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)


def download_osm_roads(bounds_wgs84: tuple, network_type: str = "drive",
                       cache_dir: str | Path = "data/osm", tag: str = ""):
    """Fetch the OSM drive network for a bounding box, with on-disk caching.

    Caching is not an optimisation here - it is what makes results reproducible.
    OSM changes daily, so an uncached pipeline silently re-labels your test set
    between runs and every comparison becomes meaningless.
    """
    import geopandas as gpd
    import osmnx as ox

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    w, s, e, n = bounds_wgs84
    key = tag or f"{w:.5f}_{s:.5f}_{e:.5f}_{n:.5f}_{network_type}"
    cache = cache_dir / f"osm_{key}.gpkg"

    if cache.exists():
        return gpd.read_file(cache, layer="edges")

    G = ox.graph_from_bbox(bbox=(w, s, e, n), network_type=network_type,
                           simplify=True, retain_all=True)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges.reset_index()[
        [c for c in ("u", "v", "key", "highway", "name", "lanes", "geometry")
         if c in edges.columns or c == "geometry"]
    ]
    edges["highway"] = edges["highway"].apply(
        lambda h: h[0] if isinstance(h, list) else h
    )
    edges.to_file(cache, layer="edges", driver="GPKG")
    (cache_dir / f"osm_{key}.meta.json").write_text(json.dumps({
        "bounds_wgs84": list(bounds_wgs84), "network_type": network_type,
        "n_edges": len(edges),
    }, indent=2))
    return edges


def rasterize_roads(
    edges,
    raster_path: str | Path,
    sensor: str | None = None,
    min_resolvable_width_m: float | None = None,
    half_width_override: float | None = None,
) -> dict:
    """Burn buffered OSM centrelines onto the exact grid of `raster_path`."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(raster_path) as src:
        transform, crs = src.transform, src.crs
        shape = (src.height, src.width)
        gsd = abs(src.transform.a)

    if min_resolvable_width_m is None:
        # A road narrower than ~1.5 px cannot be learned from this sensor.
        min_resolvable_width_m = 1.5 * (SENSOR_GSD_M.get(sensor, gsd))

    gdf = edges.to_crs(crs) if edges.crs != crs else edges.copy()

    def half_width(cls):
        if half_width_override is not None:
            return half_width_override
        return ROAD_HALF_WIDTH_M.get(cls, DEFAULT_HALF_WIDTH_M)

    gdf["half_w"] = gdf.get("highway", "road").apply(half_width)
    gdf["full_w"] = gdf["half_w"] * 2

    dropped = gdf[gdf["full_w"] < min_resolvable_width_m]
    kept = gdf[gdf["full_w"] >= min_resolvable_width_m]

    if kept.empty:
        warnings.warn("no OSM roads wide enough to resolve at this GSD")
        return {"mask": np.zeros(shape, bool), "kept": 0,
                "dropped": len(dropped), "gsd_m": gsd}

    shapes = [(geom.buffer(hw), 1) for geom, hw in zip(kept.geometry, kept["half_w"])]
    mask = rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, dtype="uint8", all_touched=True).astype(bool)

    return {
        "mask": mask,
        "kept": int(len(kept)),
        "dropped": int(len(dropped)),
        "dropped_classes": (dropped["highway"].value_counts().to_dict()
                            if len(dropped) else {}),
        "gsd_m": float(gsd),
        "min_resolvable_width_m": float(min_resolvable_width_m),
        "road_pixel_fraction": float(mask.mean()),
    }


def build_ground_truth(raster_path: str | Path, out_dir: str | Path,
                       sensor: str | None = None, network_type: str = "drive",
                       cache_dir: str | Path = "data/osm") -> dict:
    """One call: GeoTIFF in, aligned mask GeoTIFF + report out."""
    import rasterio

    raster_path, out_dir = Path(raster_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = raster_bounds_wgs84(raster_path)
    edges = download_osm_roads(bounds, network_type, cache_dir, tag=raster_path.stem)
    res = rasterize_roads(edges, raster_path, sensor)

    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")

    mask_path = out_dir / f"{raster_path.stem}_roadmask.tif"
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(res["mask"].astype("uint8") * 255, 1)

    report = {k: v for k, v in res.items() if k != "mask"}
    report.update(source_raster=str(raster_path), mask=str(mask_path),
                  bounds_wgs84=list(bounds))
    (out_dir / f"{raster_path.stem}_gt_report.json").write_text(
        json.dumps(report, indent=2))

    if res["road_pixel_fraction"] < 0.005:
        warnings.warn(
            f"only {res['road_pixel_fraction']:.3%} road pixels - check CRS "
            "alignment and that the bbox actually covers a built-up area"
        )
    return report


def osm_reference_graph(edges, crs=None):
    """Build a networkx reference graph from OSM edges for APLS scoring.

    Node 'pos' is (y, x) in the projected CRS so it is directly comparable with
    the predicted graph after georeferencing (see graph/georef.py).
    """
    import networkx as nx

    gdf = edges.to_crs(crs) if (crs is not None and edges.crs != crs) else edges
    G = nx.Graph()
    node_ids: dict = {}

    def nid(pt):
        key = (round(pt[0], 2), round(pt[1], 2))
        if key not in node_ids:
            node_ids[key] = len(node_ids) + 1
            G.add_node(node_ids[key], pos=(key[1], key[0]), kind="osm")
        return node_ids[key]

    for geom, cls in zip(gdf.geometry, gdf.get("highway", ["road"] * len(gdf))):
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        u = nid(coords[0])
        v = nid(coords[-1])
        if u == v:
            continue
        length = float(geom.length)
        if G.has_edge(u, v) and G[u][v]["length_m"] <= length:
            continue
        G.add_edge(u, v, length_m=length, length_px=length,
                   kind="observed", highway=cls,
                   pts=[(c[1], c[0]) for c in coords])
    return G
