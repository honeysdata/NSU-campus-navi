"""Build a real Korean sample dataset from the downloaded OSM extract.

The source area is Namseoul University, Cheonan, South Korea. The script
creates one label mask per imagery tile. The imagery itself is fetched from
the Esri World Imagery tile endpoint at runtime; check the provider terms
before redistribution or production use.
"""
from __future__ import annotations

import json
import math
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OSM = ROOT / "data/raw/nsu_map.osm"
OUT = ROOT / "data/datasets/namseoul_university"
Z = 18
TILE_SIZE = 256
HIGHWAYS = {"footway", "path", "pedestrian", "steps", "service", "track", "cycleway"}


def world_px(lon, lat):
    n = 2 ** Z
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


def load_osm():
    root = ET.parse(OSM).getroot()
    nodes = {int(n.attrib["id"]): (float(n.attrib["lon"]), float(n.attrib["lat"])) for n in root.findall("node")}
    ways = []
    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if tags.get("highway") not in HIGHWAYS:
            continue
        refs = [int(n.attrib["ref"]) for n in way.findall("nd")]
        coords = [nodes[r] for r in refs if r in nodes]
        if len(coords) >= 2:
            ways.append((int(way.attrib["id"]), tags, refs, coords))
    return nodes, ways


def draw_line(mask, x0, y0, x1, y1, width=5):
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        x = round(x0 + (x1 - x0) * i / steps)
        y = round(y0 + (y1 - y0) * i / steps)
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        mask[(xx - x) ** 2 + (yy - y) ** 2 <= width ** 2] = 1


def write_pgm(path, arr):
    path.write_bytes(f"P5\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode() + (arr * 255).astype(np.uint8).tobytes())


def write_geojson(ways):
    features = []
    for wid, tags, refs, coords in ways:
        features.append({"type": "Feature", "properties": {"osm_way_id": wid, **tags}, "geometry": {"type": "LineString", "coordinates": coords}})
    (OUT / "osm_walkways.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if not OSM.exists():
        raise SystemExit("nsu_map.osm이 없습니다. 먼저 OSM API에서 다운로드하세요.")
    nodes, ways = load_osm()
    OUT.mkdir(exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    (OUT / "masks").mkdir(exist_ok=True)
    write_geojson(ways)

    pixels = [world_px(lon, lat) for _, _, _, coords in ways for lon, lat in coords]
    minx, maxx = min(p[0] for p in pixels), max(p[0] for p in pixels)
    miny, maxy = min(p[1] for p in pixels), max(p[1] for p in pixels)
    tx0, tx1 = int(minx // TILE_SIZE), int(maxx // TILE_SIZE)
    ty0, ty1 = int(miny // TILE_SIZE), int(maxy // TILE_SIZE)
    tile_count = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            name = f"{Z}_{tx}_{ty}"
            mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
            for _, _, _, coords in ways:
                for a, b in zip(coords, coords[1:]):
                    ax, ay = world_px(*a); bx, by = world_px(*b)
                    draw_line(mask, round(ax - tx * TILE_SIZE), round(ay - ty * TILE_SIZE), round(bx - tx * TILE_SIZE), round(by - ty * TILE_SIZE))
            if mask.any():
                write_pgm(OUT / "masks" / f"{name}.pgm", mask)
                url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{ty}/{tx}"
                try:
                    image_path = OUT / "images" / f"{name}.jpg"
                    if not image_path.exists():
                        request = urllib.request.Request(url, headers={"User-Agent": "NSU-mentoring-pilot/1.0"})
                        with urllib.request.urlopen(request, timeout=8) as response:
                            image_path.write_bytes(response.read())
                except Exception as exc:
                    print(f"imagery download failed for {name}: {exc}")
                tile_count += 1
    meta = {"area": "Namseoul University, Cheonan, South Korea", "z": Z, "ways": len(ways), "tiles": tile_count, "classes": {"0": "background", "1": "OSM walkable corridor buffer"}, "note": "The label is a pseudo-label from OSM centerlines; verify manually before training."}
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
