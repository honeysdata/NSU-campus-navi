# 실제 서비스형 구조

```text
web_demo/       사용자 길찾기 + 관리자 라벨 검수
backend/        FastAPI: 캠퍼스·경로 API
scripts/        데이터 구축·학습·추론·그래프 변환 실행 코드
db/             PostGIS schema와 migration
data/datasets/namseoul_university/ 학습 데이터·모델·예측 결과
```

## 현재 전환 단계

`backend/app.py`는 PostgreSQL/PostGIS의 `published` 그래프만 조회합니다. `data/campuses.json`과 초기 그래프 JSON은 초기 적재용 원본이며, 실행 중 API의 저장소가 아닙니다. 남서울대학교와 서울대학교 관악 캠퍼스가 PostGIS에 적재됩니다.

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
