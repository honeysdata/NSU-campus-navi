from __future__ import annotations
import math, os
from psycopg_pool import ConnectionPool

class PostGISRepository:
    def __init__(self):
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required")
        self.pool = ConnectionPool(
            database_url,
            min_size=int(os.getenv("DB_POOL_MIN_SIZE", "2")),
            max_size=int(os.getenv("DB_POOL_MAX_SIZE", "10")),
            timeout=float(os.getenv("DB_POOL_TIMEOUT_SEC", "10")),
            max_waiting=int(os.getenv("DB_POOL_MAX_WAITING", "50")),
            max_lifetime=float(os.getenv("DB_POOL_MAX_LIFETIME_SEC", "1800")),
            reconnect_timeout=float(os.getenv("DB_POOL_RECONNECT_TIMEOUT_SEC", "30")),
            open=False,
        )

    def open(self): self.pool.open()
    def close(self): self.pool.close()

    def campuses(self):
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT campus_id,name,ST_Y(center),ST_X(center) FROM campuses ORDER BY name")
            return [{"campus_id": r[0], "name": r[1], "center": [r[2], r[3]] if r[2] is not None else None} for r in cur.fetchall()]

    def graph_info(self, campus_id):
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT version_id FROM nav_graph_versions WHERE campus_id=%s AND status='published'", (campus_id,)); row=cur.fetchone()
            if not row: return None
            version=row[0]
            cur.execute("SELECT count(*) FROM nav_nodes WHERE campus_id=%s AND version_id=%s", (campus_id,version)); nodes=cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM nav_edges WHERE campus_id=%s AND version_id=%s", (campus_id,version)); edges=cur.fetchone()[0]
            return {"version_id":version,"node_count":nodes,"edge_count":edges}

    def nearest(self, campus_id, lon, lat):
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT n.node_id,ST_X(n.geom),ST_Y(n.geom) FROM nav_nodes n JOIN nav_graph_versions v ON v.version_id=n.version_id WHERE n.campus_id=%s AND v.status='published' ORDER BY n.geom <-> ST_SetSRID(ST_Point(%s,%s),4326) LIMIT 1", (campus_id,lon,lat)); row=cur.fetchone()
            if not row: return None
            return {"id":row[0],"lon":row[1],"lat":row[2]}

    def graph(self, campus_id):
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT v.version_id FROM nav_graph_versions v WHERE v.campus_id=%s AND v.status='published'", (campus_id,)); row=cur.fetchone()
            if not row: return {}, []
            version=row[0]
            cur.execute("SELECT node_id,ST_X(geom),ST_Y(geom) FROM nav_nodes WHERE campus_id=%s AND version_id=%s", (campus_id,version))
            nodes={r[0]:{"id":r[0],"lon":r[1],"lat":r[2]} for r in cur.fetchall()}
            cur.execute("SELECT source,target,length_m FROM nav_edges WHERE campus_id=%s AND version_id=%s", (campus_id,version))
            return nodes, cur.fetchall()

    def route(self, campus_id, start, destination):
        source=self.nearest(campus_id,start["lon"],start["lat"]); target=self.nearest(campus_id,destination["lon"],destination["lat"])
        if not source or not target: return None
        nodes, edges=self.graph(campus_id); adj={i:[] for i in nodes}
        for a,b,w in edges: adj[a].append((b,w)); adj[b].append((a,w))
        dist={i:float('inf') for i in nodes}; prev={}; dist[source['id']]=0; left=set(nodes)
        while left:
            u=min(left,key=lambda x:dist[x]); left.remove(u)
            if dist[u]==float('inf') or u==target['id']: break
            for v,w in adj[u]:
                if dist[u]+w<dist[v]: dist[v]=dist[u]+w; prev[v]=u
        if dist[target['id']]==float('inf'): return None
        path=[]; u=target['id']
        while True:
            path.insert(0,nodes[u])
            if u==source['id']: break
            u=prev[u]
        return {"distance_m":round(dist[target['id']],1),"source_node":source['id'],"target_node":target['id'],"geometry":{"type":"LineString","coordinates":[[n['lon'],n['lat']] for n in path]},"node_count":len(path)}
