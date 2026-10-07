CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS campuses (
  campus_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  srid INTEGER NOT NULL DEFAULT 4326,
  boundary geometry(MultiPolygon, 4326),
  center geometry(Point, 4326),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nav_graph_versions (
  version_id BIGSERIAL PRIMARY KEY,
  campus_id TEXT NOT NULL,
  source TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('draft','reviewed','published','archived')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS one_published_graph_per_campus
ON nav_graph_versions (campus_id) WHERE status = 'published';

CREATE TABLE IF NOT EXISTS nav_nodes (
  node_id BIGINT,
  version_id BIGINT REFERENCES nav_graph_versions(version_id),
  campus_id TEXT NOT NULL,
  node_type TEXT NOT NULL,
  confidence REAL,
  geom geometry(Point, 4326) NOT NULL,
  PRIMARY KEY (node_id, version_id)
);

CREATE TABLE IF NOT EXISTS nav_edges (
  edge_id BIGINT,
  version_id BIGINT REFERENCES nav_graph_versions(version_id),
  campus_id TEXT NOT NULL,
  source BIGINT NOT NULL,
  target BIGINT NOT NULL,
  length_m REAL NOT NULL,
  stairs BOOLEAN DEFAULT FALSE,
  accessible BOOLEAN DEFAULT TRUE,
  confidence REAL,
  geom geometry(LineString, 4326) NOT NULL,
  PRIMARY KEY (edge_id, version_id)
);

CREATE INDEX IF NOT EXISTS nav_nodes_geom_idx ON nav_nodes USING GIST (geom);
CREATE INDEX IF NOT EXISTS nav_edges_geom_idx ON nav_edges USING GIST (geom);
CREATE INDEX IF NOT EXISTS nav_nodes_campus_version_idx ON nav_nodes (campus_id, version_id);
CREATE INDEX IF NOT EXISTS nav_edges_campus_version_idx ON nav_edges (campus_id, version_id);

CREATE TABLE IF NOT EXISTS label_assets (
  asset_id BIGSERIAL PRIMARY KEY,
  campus_id TEXT NOT NULL REFERENCES campuses(campus_id),
  version_id BIGINT REFERENCES nav_graph_versions(version_id),
  source_uri TEXT NOT NULL,
  label_type TEXT NOT NULL CHECK (label_type IN ('osm','human_review','model_prediction')),
  status TEXT NOT NULL CHECK (status IN ('draft','reviewed','approved','archived')),
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS review_events (
  review_id BIGSERIAL PRIMARY KEY,
  campus_id TEXT NOT NULL REFERENCES campuses(campus_id),
  version_id BIGINT REFERENCES nav_graph_versions(version_id),
  reviewer TEXT,
  action TEXT NOT NULL CHECK (action IN ('add','delete','move','approve','reject')),
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW published_nav_edges AS
SELECT e.*
FROM nav_edges e
JOIN nav_graph_versions v ON v.version_id = e.version_id
WHERE v.status = 'published';
