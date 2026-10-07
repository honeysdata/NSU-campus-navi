# 다중 캠퍼스 파일럿

## 등록된 캠퍼스

`data/campuses.json`이 캠퍼스 레지스트리입니다.

- `namseoul`: 남서울대학교 천안 캠퍼스 — 영상·라벨·U-Net 파일럿 포함
- `snu-gwanak`: 서울대학교 관악 캠퍼스 — 실제 OSM 그래프 파일럿

각 캠퍼스는 별도의 그래프 파일을 가질 수 있지만 API는 동일합니다.

```text
POST /api/routes
campus_id = namseoul

POST /api/routes
campus_id = snu-gwanak
```

## 새 캠퍼스 추가

```bash
uv run build_graph_json.py \
  --source new-campus.osm \
  --output data/new-campus.json
```

그 다음 `data/campuses.json`에 다음 항목을 추가합니다.

```json
{
  "campus_id": "new-campus",
  "name": "새 캠퍼스",
  "data": "../data/new-campus.json",
  "center": [위도, 경도]
}
```

운영 환경에서는 이 파일을 DB의 `campuses` 테이블로 옮기고, 그래프 파일은 PostGIS의 `nav_nodes`, `nav_edges`로 적재합니다.
