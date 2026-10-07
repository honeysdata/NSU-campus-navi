"""NSU 진취동 3기 - segmentation to navigation graph mini pilot.

Run:
    python3 pilot.py

The demo creates a tiny synthetic campus mask, extracts a centerline graph,
and writes GeoJSON, SQL, and SVG outputs. The synthetic mask stands in for
the output of a U-Net/DeepLab model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

OUT = Path(__file__).parent / "pilot_output"
H = W = 160


def disk_distance_to_segment(y, x, y1, x1, y2, x2):
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    t = np.clip(((x - x1) * vx + (y - y1) * vy) / denom, 0, 1)
    px, py = x1 + t * vx, y1 + t * vy
    return np.sqrt((x - px) ** 2 + (y - py) ** 2)


def make_synthetic_mask() -> np.ndarray:
    """Make a walkable-area mask resembling paths around campus buildings."""
    yy, xx = np.mgrid[0:H, 0:W]
    segments = [
        (20, 80, 140, 80),       # main east-west path
        (80, 20, 80, 140),       # main north-south path
        (80, 45, 45, 25),        # northwest branch
        (80, 115, 120, 140),      # southeast branch
        (45, 80, 30, 55),         # upper spur
    ]
    mask = np.zeros((H, W), dtype=np.uint8)
    for y1, x1, y2, x2 in segments:
        mask[disk_distance_to_segment(yy, xx, y1, x1, y2, x2) <= 5] = 1
    # A small square plaza demonstrates that segmentation outputs are areas.
    mask[67:94, 67:94] = 1
    return mask


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Small dependency-free Zhang-Suen skeletonization implementation."""
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            remove = []
            for y in range(1, img.shape[0] - 1):
                for x in range(1, img.shape[1] - 1):
                    if img[y, x] == 0:
                        continue
                    p = [img[y-1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1],
                         img[y+1, x], img[y+1, x-1], img[y, x-1], img[y-1, x-1]]
                    neighbors = sum(p)
                    transitions = sum(p[i] == 0 and p[(i+1) % 8] == 1 for i in range(8))
                    if not (2 <= neighbors <= 6 and transitions == 1):
                        continue
                    if step == 0:
                        ok = p[0] * p[2] * p[4] == 0 and p[2] * p[4] * p[6] == 0
                    else:
                        ok = p[0] * p[2] * p[6] == 0 and p[0] * p[4] * p[6] == 0
                    if ok:
                        remove.append((y, x))
            if remove:
                changed = True
                for y, x in remove:
                    img[y, x] = 0
    return img


def neighbors(p: Tuple[int, int], skel: np.ndarray) -> List[Tuple[int, int]]:
    y, x = p
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and skel[y + dy, x + dx]:
                result.append((y + dy, x + dx))
    return result


def extract_graph(skel: np.ndarray):
    pixels = {(y, x) for y, x in zip(*np.where(skel))}
    key_pixels = {p for p in pixels if len(neighbors(p, skel)) != 2}
    # Cluster adjacent key pixels into one logical node.
    nodes: List[Tuple[int, int]] = []
    unvisited = set(key_pixels)
    while unvisited:
        seed = unvisited.pop(); group = {seed}; stack = [seed]
        while stack:
            p = stack.pop()
            for q in neighbors(p, skel):
                if q in unvisited:
                    unvisited.remove(q); group.add(q); stack.append(q)
        y = round(sum(p[0] for p in group) / len(group))
        x = round(sum(p[1] for p in group) / len(group))
        nodes.append((y, x))
    node_id = {p: i + 1 for i, p in enumerate(nodes)}

    def nearest_node(p):
        return min(nodes, key=lambda n: (n[0] - p[0]) ** 2 + (n[1] - p[1]) ** 2)

    edges = set()
    for node in nodes:
        for start in neighbors(node, skel):
            prev, cur = node, start
            path = [node, cur]
            for _ in range(H * W):
                if cur in node_id and cur != node:
                    a, b = node_id[node], node_id[cur]
                    edges.add((min(a, b), max(a, b), tuple(path)))
                    break
                nxt = [q for q in neighbors(cur, skel) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
    return nodes, list(edges)


def xy(y: int, x: int) -> List[float]:
    # Demo-only local georeferencing. Replace with the source raster transform.
    return [127.1000 + x * 0.00001, 37.5000 + (H - y) * 0.00001]


def write_pgm(path: Path, arr: np.ndarray):
    path.write_bytes(f"P5\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode() + (arr * 255).astype(np.uint8).tobytes())


def main():
    OUT.mkdir(exist_ok=True)
    mask = make_synthetic_mask()
    skel = zhang_suen_thinning(mask)
    write_pgm(OUT / "synthetic_walkable_mask.pgm", mask)
    write_pgm(OUT / "skeleton.pgm", skel)

    nodes, edges = extract_graph(skel)
    node_features = []
    for i, (y, x) in enumerate(nodes, 1):
        node_features.append({"type": "Feature", "properties": {"node_id": i, "node_type": "junction", "confidence": 1.0}, "geometry": {"type": "Point", "coordinates": xy(y, x)}})
    edge_features = []
    for i, (a, b, path) in enumerate(sorted(edges, key=lambda e: (e[0], e[1], len(e[2]))), 1):
        coords = [xy(y, x) for y, x in path]
        length = sum(((coords[j][0] - coords[j-1][0]) ** 2 + (coords[j][1] - coords[j-1][1]) ** 2) ** 0.5 for j in range(1, len(coords))) * 111_000
        edge_features.append({"type": "Feature", "properties": {"edge_id": i, "source": a, "target": b, "length_m": round(length, 2), "walkable": True, "confidence": 1.0}, "geometry": {"type": "LineString", "coordinates": coords}})
    for name, features in (("nodes", node_features), ("edges", edge_features)):
        (OUT / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")

    sql = ["CREATE EXTENSION IF NOT EXISTS postgis;", "", "CREATE TABLE IF NOT EXISTS nav_nodes (node_id BIGINT PRIMARY KEY, node_type TEXT, confidence REAL, geom geometry(Point, 4326));", "CREATE TABLE IF NOT EXISTS nav_edges (edge_id BIGINT PRIMARY KEY, source BIGINT REFERENCES nav_nodes(node_id), target BIGINT REFERENCES nav_nodes(node_id), length_m REAL, walkable BOOLEAN, confidence REAL, geom geometry(LineString, 4326));", "CREATE INDEX IF NOT EXISTS nav_nodes_geom_idx ON nav_nodes USING GIST (geom);", "CREATE INDEX IF NOT EXISTS nav_edges_geom_idx ON nav_edges USING GIST (geom);", ""]
    for f in node_features:
        p = f["properties"]; c = f["geometry"]["coordinates"]
        sql.append(f"INSERT INTO nav_nodes VALUES ({p['node_id']}, '{p['node_type']}', {p['confidence']}, ST_SetSRID(ST_Point({c[0]}, {c[1]}), 4326));")
    for f in edge_features:
        p = f["properties"]; coords = f["geometry"]["coordinates"]
        wkt = ", ".join(f"{x} {y}" for x, y in coords)
        sql.append(f"INSERT INTO nav_edges VALUES ({p['edge_id']}, {p['source']}, {p['target']}, {p['length_m']}, TRUE, {p['confidence']}, ST_SetSRID(ST_GeomFromText('LINESTRING({wkt})'), 4326));")
    (OUT / "load_postgis.sql").write_text("\n".join(sql) + "\n", encoding="utf-8")

    print(f"nodes={len(nodes)} edges={len(edges)}")
    print(f"outputs: {OUT}")


if __name__ == "__main__":
    main()
