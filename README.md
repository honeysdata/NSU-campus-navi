# NSU 진취동 3기: segmentation → graph → PostGIS 파일럿

## uv 환경에서 실행

Python 3.11을 사용하는 것을 권장합니다.

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra ml --extra server
```

## 실행

```bash
uv run pilot.py
```

`pilot_output/`에 다음 결과가 생성됩니다.

- `synthetic_walkable_mask.pgm`: AI 세그멘테이션 결과를 흉내 낸 보행 영역 마스크
- `skeleton.pgm`: 마스크에서 추출한 중심선
- `nodes.geojson`: 분기점·끝점
- `edges.geojson`: 그래프 간선
- `load_postgis.sql`: PostGIS 테이블·인덱스·데이터 적재 SQL

## 실제 남서울대학교 데이터 시연

```bash
uv run build_nsu_dataset.py
uv run train_unet.py
uv run demo.py
```

웹 시연:

```bash
uv run python3 -m http.server 8765 --directory .
```

브라우저에서 `http://127.0.0.1:8765/web_demo/`를 엽니다.

시연 결과는 다음 파일에서 확인합니다.

- `nsu_real_dataset/demo_overlay.png`: 항공영상 위에 모델 예측 마스크를 표시한 결과
- `nsu_real_dataset/demo_predicted_mask.png`: 모델의 이진 예측 마스크

## 핵심 개념

1. 실제 모델은 `synthetic_walkable_mask`를 생성하는 자리에 들어간다.
2. 마스크는 보행 영역이고, 길찾기 그래프가 아니다.
3. 중심선화 후 연결 차수가 달라지는 점을 노드로, 노드 사이 선을 간선으로 만든다.
4. GeoJSON은 확인·교환용이고, 서비스에서는 PostGIS에 적재한다.
5. 실제 데이터에서는 픽셀 좌표 대신 원본 래스터의 affine transform을 사용해야 한다.

## 실제 데이터로 확장할 때 바꿀 부분

- `make_synthetic_mask()` → U-Net/DeepLab 추론 결과
- `zhang_suen_thinning()` → `scikit-image` skeletonize
- `xy()` → Rasterio transform으로 픽셀을 지도 좌표로 변환
- 수동 검수 및 confidence 기반 승인 단계 추가
