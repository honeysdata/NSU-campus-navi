"""Load all registered campus graph JSON files into PostGIS."""
import json, os, math
from pathlib import Path
import psycopg

ROOT = Path(__file__).resolve().parents[1]
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL is required")

def main():
    registry = json.loads((ROOT / "data/campuses.json").read_text(encoding="utf-8"))
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for campus in registry:
                campus_id = campus["campus_id"]
                data = json.loads((ROOT / campus["data"].removeprefix("../")).read_text(encoding="utf-8"))
                cur.execute("INSERT INTO campuses(campus_id,name,center) VALUES (%s,%s,ST_SetSRID(ST_Point(%s,%s),4326)) ON CONFLICT (campus_id) DO UPDATE SET name=EXCLUDED.name,center=EXCLUDED.center", (campus_id, campus["name"], campus["center"][1], campus["center"][0]))
                cur.execute("UPDATE nav_graph_versions SET status='archived' WHERE campus_id=%s AND status='published'", (campus_id,))
                cur.execute("INSERT INTO nav_graph_versions(campus_id,source,status) VALUES (%s,%s,'published') RETURNING version_id", (campus_id, "osm_graph_json"))
                version_id = cur.fetchone()[0]
                for n in data["nodes"]:
                    cur.execute("INSERT INTO nav_nodes(node_id,version_id,campus_id,node_type,confidence,geom) VALUES (%s,%s,%s,'junction',1.0,ST_SetSRID(ST_Point(%s,%s),4326))", (n["id"], version_id, campus_id, n["lon"], n["lat"]))
                for i, e in enumerate(data["edges"], 1):
                    coords = e["coords"]
                    wkt = "LINESTRING(" + ",".join(f"{x} {y}" for x, y in coords) + ")"
                    x1, y1 = coords[0]; x2, y2 = coords[-1]
                    length_m = math.hypot((y2-y1)*111000, (x2-x1)*89000)
                    cur.execute("INSERT INTO nav_edges(edge_id,version_id,campus_id,source,target,length_m,confidence,geom) VALUES (%s,%s,%s,%s,%s,%s,1.0,ST_SetSRID(ST_GeomFromText(%s),4326))", (i, version_id, campus_id, e["source"], e["target"], length_m, wkt))
        conn.commit()
    print(f"loaded {len(registry)} campuses into PostGIS")

if __name__ == "__main__": main()
