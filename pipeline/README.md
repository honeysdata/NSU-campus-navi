# Pipeline modules

운영 환경에서는 아래 작업을 API 서버와 분리된 배치/worker로 실행합니다.

```text
collect.py       항공영상·OSM 수집
prepare_labels.py사람 검수 GeoJSON → 학습 마스크
train.py         U-Net/DeepLab 학습
infer.py         새 영상 추론
vectorize.py     마스크 → 중심선 → 노드·간선
publish.py       검수 승인 결과를 published 버전으로 반영
```

현재 파일럿에서는 기존 스크립트가 이 역할을 담당합니다.

- `build_nsu_dataset.py` = collect + prepare_labels
- `train_unet.py` = train
- `demo.py` = infer
- `make_web_data.py` = 임시 graph export
