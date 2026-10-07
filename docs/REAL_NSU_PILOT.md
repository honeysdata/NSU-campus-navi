# 실제 한국 데이터 파일럿: 남서울대학교

## 확보한 데이터

- 지역: 충청남도 천안시 서북구 성환읍 남서울대학교 주변
- 원본 벡터: OpenStreetMap 공식 API에서 받은 `data/raw/nsu_map.osm`
- 보행로 필터: `footway`, `path`, `pedestrian`, `steps`, `service`, `track`, `cycleway`
- 영상 입력: Esri World Imagery tile, Web Mercator zoom 18
- 라벨: OSM 보행로 중심선을 픽셀 폭 5로 확장한 pseudo-label

OSM 라벨은 실제 정답이 아니라 기존 지도 데이터를 이용한 초기 라벨입니다. 검수자는 항공영상과 겹쳐 보면서 잘못된 길, 누락된 길, 차량용 서비스도로를 수정해야 합니다.

## 실행 결과

```bash
python3 build_nsu_dataset.py
```

현재 `data/datasets/namseoul_university/`에는 남서울대학교 주변의 실제 영상 타일 45장과 동일 위치의 마스크 45장이 있습니다.

```text
data/datasets/namseoul_university/
├── images/                  # 256×256 실제 항공영상 타일
├── masks/                   # 동일 타일의 OSM 기반 보행로 라벨
├── osm_walkways.geojson     # 원본 보행로 벡터
└── metadata.json
```

## 모델 학습

PyTorch가 설치된 환경에서:

```bash
pip install -r requirements-ml.txt
python3 train_unet.py
```

이 실험의 목적은 높은 정확도보다 “실제 한국 캠퍼스 이미지 → 보행로 마스크”의 전체 학습 흐름을 확인하는 것입니다.

## 반드시 할 검수

1. 영상과 마스크가 같은 위치를 가리키는지 확인
2. OSM에 없는 새 보행로가 있는지 확인
3. `service` 도로가 차량 진입로인지 보행 가능로인지 확인
4. 학습·검증 영역을 인접 타일이 아니라 공간적으로 나누기
5. 최종적으로 사람이 수정한 라벨을 별도 보관하기

## 라이선스 주의

OSM 데이터는 OpenStreetMap의 라이선스와 attribution 조건을 따라야 합니다. Esri 영상 타일은 제공자의 이용약관을 확인한 뒤 학습·배포 범위를 결정해야 합니다. 이 저장소의 데이터는 멘토링용 파일럿과 검증용으로만 사용하고, 서비스 배포 전에는 정식 사용 가능 영상 또는 기관 제공 정사영상을 확보하는 것을 권장합니다.
