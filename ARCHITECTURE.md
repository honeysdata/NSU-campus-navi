# 실제 서비스형 구조

```text
web_demo/       사용자 길찾기 + 관리자 라벨 검수
backend/        FastAPI: 캠퍼스·경로 API
pipeline/       수집·라벨·학습·추론·벡터화 worker
db/             PostGIS schema와 migration
nsu_real_dataset/ 파일럿 산출물
```

## 현재 전환 단계

`backend/app.py`는 PostgreSQL/PostGIS의 `published` 그래프만 조회합니다. `data/campuses.json`과 그래프 JSON은 초기 적재용 원본이며, 실행 중 API의 저장소가 아닙니다. 남서울대학교와 서울대학교 관악 캠퍼스가 PostGIS에 적재됩니다.

실행:

```bash
uv sync --extra ml --extra server
uv run uvicorn backend.app:app --reload --port 8000
```

API 문서:

```text
http://127.0.0.1:8000/docs
```

로컬 PostGIS:

```bash
docker compose up -d postgis
```
