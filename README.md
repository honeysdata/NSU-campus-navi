# NSU Campus Navi

항공영상·OSM 데이터를 이용해 캠퍼스 보행로를 세그멘테이션하고, 길찾기용 그래프와 PostGIS API로 연결하는 파일럿 프로젝트입니다.

## 디렉토리 안내

```text
backend/                         FastAPI 길찾기 API
db/                              PostGIS 스키마와 그래프 적재 코드
web_demo/                        계획·AI 파이프라인·라벨 검수·길찾기 화면
scripts/                         데이터 구축·학습·추론·그래프 변환 실행 코드
data/raw/                        OSM 원본 파일
data/datasets/namseoul_university/ 남서울대 항공 타일·마스크·모델 결과
data/campuses.json               캠퍼스 레지스트리
outputs/graph_pilot/             마스크→중심선→노드·간선 변환 결과
docs/                            구조·파이프라인·다중 캠퍼스 문서
```

## 1. 환경 설치

```bash
cd /Users/song/PROJECT/gis
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra ml --extra server
```

## 2. 그래프 변환 파일럿

AI가 보행 마스크를 만들었다고 가정하고, 마스크를 중심선·노드·간선으로 변환합니다.

```bash
uv run python scripts/pilot.py
```

결과는 `outputs/graph_pilot/`에 저장됩니다.

## 3. 실제 남서울대 데이터 구축

OSM 원본을 사용해 항공영상 타일과 학습용 마스크를 생성합니다.

```bash
uv run python scripts/build_nsu_dataset.py
```

## 4. 모델 학습 및 비교

```bash
uv run python scripts/train_unet.py
uv run python scripts/train_mobilenet_full.py
uv run python scripts/compare_models.py
```

각 명령은 기본 U-Net 학습, ImageNet 사전학습 MobileNetV3 기반 모델 학습, 두 모델의 결과 비교를 수행합니다.

## 5. 단일 이미지 추론

```bash
uv run python scripts/demo.py
```

학습된 체크포인트로 예측 마스크와 항공영상 오버레이를 생성합니다.

## 6. 웹 데모 실행

`file://`로 열지 말고 HTTP 서버로 실행해야 JSON·지도·이미지를 정상적으로 불러옵니다.

```bash
python3 -m http.server 8766 --bind 127.0.0.1
```

브라우저에서 http://127.0.0.1:8766/web_demo/ 를 엽니다.

- 계획: 서비스 플로우와 3주 계획
- 데이터·AI 파이프라인: 입력·라벨·예측·그래프 단계와 모델 비교
- 라벨 검수: OSM 라벨을 지도 위에서 수정하고 GeoJSON으로 다운로드
- 길찾기 서비스: 캠퍼스 선택, 목적지 핀 지정, 경로 조회

## 7. PostGIS와 API 실행

`.env`에 DB 설정이 필요합니다.

```bash
cp .env.example .env
docker compose up --build
```

API 문서는 http://127.0.0.1:8000/docs 에서 확인합니다.

- `GET /health`: DB 연결과 저장소 상태
- `GET /api/campuses`: 등록 캠퍼스 목록
- `GET /api/campuses/{campus_id}`: 캠퍼스 그래프 정보
- `POST /api/routes`: 시작점·목적지 사이 도보 경로

## 8. 현재 파일럿 범위

현재 API는 PostGIS에 적재된 OSM 기반 그래프를 사용합니다. AI 예측 마스크에서 노드·간선을 자동 생성해 PostGIS에 반영하는 부분은 다음 고도화 단계입니다. 학습 라벨은 OSM 기반 pseudo-label이므로 라벨 검수와 실제 GIS 폭 기반 Buffer가 필요합니다.

자세한 내용은 `docs/ARCHITECTURE.md`, `docs/PIPELINE.md`, `docs/MULTI_CAMPUS.md`를 참고하세요.
