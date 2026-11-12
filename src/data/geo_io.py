"""
Geospatial I/O - rasterio/GDAL wrappers for real satellite products.

Everything downstream works on plain numpy arrays; this module is the only place
that knows about CRS, transforms and band ordering. Keeping that boundary sharp
is what lets the entire graph/analytics stack be tested without GDAL installed.

Sensor notes that matter in practice:
  Sentinel-2   L2A, 10 m for B2/B3/B4/B8. Bands live in separate JP2 files at
               different resolutions - B8 (NIR) is 10 m, but B11/B12 are 20 m and
               must be resampled before stacking.
  LISS-IV      5.8 m, 3 bands (G/R/NIR) - note there is NO blue band, so an
               ImageNet-pretrained RGB stem must be remapped, not fed naively.
  Cartosat-3   0.25 m PAN. At this GSD a single lane is ~14 px wide, so tile
               sizes and healing gap thresholds must be scaled accordingly:
               a 60 px gap is 600 m on Sentinel-2 but only 15 m on Cartosat-3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SENSOR_BANDS = {
    "sentinel2": {"gsd_m": 10.0, "bands": ["B2", "B3", "B4", "B8"],
                  "rgb": (2, 1, 0), "nir": 3, "bit_depth": 12, "scale": 10000},
    "liss4": {"gsd_m": 5.8, "bands": ["G", "R", "NIR"],
              "rgb": (1, 0, 2), "nir": 2, "bit_depth": 10, "scale": 1023},
    "cartosat3": {"gsd_m": 0.25, "bands": ["B", "G", "R", "NIR"],
                  "rgb": (2, 1, 0), "nir": 3, "bit_depth": 11, "scale": 2047},
}


def read_raster(path: str | Path, bands: list[int] | None = None,
                window: tuple | None = None) -> dict:
    """Read a GeoTIFF into (H, W, C) float32 plus its georeferencing.

    `window` is (row_off, col_off, height, width) - use it for scenes too large
    to hold in memory; rasterio reads only those blocks from disk.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        win = Window(window[1], window[0], window[3], window[2]) if window else None
        idx = bands or list(range(1, src.count + 1))
        arr = src.read(idx, window=win).astype(np.float32)
        transform = src.window_transform(win) if win else src.transform
        return {
            "array": np.transpose(arr, (1, 2, 0)),
            "transform": transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "gsd_m": abs(transform.a),
            "shape": arr.shape[1:],
            "descriptions": src.descriptions,
        }


def write_raster(path: str | Path, array: np.ndarray, transform, crs,
                 dtype: str = "uint8", nodata=0) -> Path:
    """Write (H, W) or (H, W, C) to a compressed, tiled GeoTIFF."""
    import rasterio

    a = array[..., None] if array.ndim == 2 else array
    a = np.transpose(a, (2, 0, 1))
    profile = {
        "driver": "GTiff", "height": a.shape[1], "width": a.shape[2],
        "count": a.shape[0], "dtype": dtype, "crs": crs, "transform": transform,
        "nodata": nodata, "compress": "deflate", "tiled": True,
        "blockxsize": 512, "blockysize": 512,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a.astype(dtype))
    return Path(path)


def to_reflectance(array: np.ndarray, sensor: str) -> np.ndarray:
    """Convert raw DN to 0-1 reflectance using the sensor's scale factor.

    Skipping this is a classic silent bug: Sentinel-2 L2A DNs run 0-10000, so
    feeding them to an ImageNet-normalized model produces activations ~40x too
    large and the model appears to fail for no visible reason.
    """
    scale = SENSOR_BANDS.get(sensor, {}).get("scale", 10000)
    return np.clip(array.astype(np.float32) / scale, 0, 1)


def select_rgb(array: np.ndarray, sensor: str) -> np.ndarray:
    """Reorder sensor bands into true-colour RGB.

    LISS-IV has no blue band; we substitute NIR-derived green so the stem still
    receives three plausible channels. Document this in any results table - it is
    a real domain shift versus RGB-pretrained weights, not a cosmetic detail.
    """
    cfg = SENSOR_BANDS.get(sensor)
    if cfg is None or array.shape[-1] < 3:
        return array[..., :3]
    if sensor == "liss4":
        g, r, nir = array[..., 0], array[..., 1], array[..., 2]
        return np.stack([r, g, np.clip(g * 1.2 - nir * 0.2, 0, 1)], axis=-1)
    idx = cfg["rgb"]
    return np.stack([array[..., i] for i in idx], axis=-1)


def compute_ndvi(array: np.ndarray, sensor: str) -> np.ndarray:
    """NDVI as an extra input channel: an explicit cue for the tree canopy that
    causes most occlusions. Cheap, and it measurably helps the aux head."""
    cfg = SENSOR_BANDS.get(sensor)
    if cfg is None or array.shape[-1] <= cfg["nir"]:
        raise ValueError(f"no NIR band available for sensor {sensor}")
    nir = array[..., cfg["nir"]]
    red = array[..., cfg["rgb"][0]]
    return (nir - red) / np.maximum(nir + red, 1e-6)


def pixel_to_geo(rows, cols, transform) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) -> (x, y) in the raster CRS, using pixel centres."""
    import rasterio

    xs, ys = rasterio.transform.xy(transform, np.asarray(rows), np.asarray(cols),
                                   offset="center")
    return np.asarray(xs), np.asarray(ys)


def geo_to_wgs84(xs, ys, crs) -> tuple[np.ndarray, np.ndarray]:
    """Project to lon/lat for Leaflet/Folium, which only speaks EPSG:4326."""
    from rasterio.warp import transform as warp_transform

    lon, lat = warp_transform(crs, "EPSG:4326", list(np.atleast_1d(xs)),
                              list(np.atleast_1d(ys)))
    return np.asarray(lon), np.asarray(lat)


def scene_iterator(path: str | Path, tile_size: int = 1024, overlap: int = 128):
    """Yield windowed reads of a scene too large for memory.

    A Sentinel-2 tile is 10980x10980x4 float32 = ~1.9 GB. Streaming windows is
    not an optimisation, it is the only way this runs on a laptop.
    """
    import rasterio

    with rasterio.open(path) as src:
        h, w = src.height, src.width
    stride = tile_size - overlap
    for r in range(0, h, stride):
        for c in range(0, w, stride):
            th, tw = min(tile_size, h - r), min(tile_size, w - c)
            if th < 32 or tw < 32:
                continue
            yield read_raster(path, window=(r, c, th, tw)), (r, c)
