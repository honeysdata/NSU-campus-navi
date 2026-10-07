# NSU Campus Navi 확인용 도식

이 문서는 웹 데모 화면에 포함하지 않는 개발·멘토링 확인용 도식입니다.

## 1. 전체 서비스 구조

```mermaid
flowchart LR
    A[항공영상·OSM 원본] --> B[데이터 구축]
    B --> C[학습 데이터셋]
    C --> D[세그멘테이션 모델]
    D --> E[보행 가능 영역 Mask]
    E --> F[후처리·중심선화]
    F --> G[노드·간선 Graph]
    G --> H[(PostGIS)]
    H --> I[FastAPI Route API]
    I --> J[길찾기 웹 서비스]
    K[라벨 검수 UI] --> L[검수 GeoJSON]
    L --> B
```

## 2. 저장소 디렉토리 관계

```mermaid
flowchart TD
    ROOT[프로젝트 루트]
    ROOT --> S[scripts/ 실행 코드]
    ROOT --> B[backend/ API 서버]
    ROOT --> DB[db/ DB 스키마·적재]
    ROOT --> WEB[web_demo/ 웹 화면]
    ROOT --> DOC[docs/ 문서·도식]
    ROOT --> DATA[data/ 데이터]
    ROOT --> OUT[outputs/ 결과]
    DATA --> RAW[data/raw/ OSM 원본]
    DATA --> DS[data/datasets/namseoul_university/]
    DS --> IMG[images/ 항공 타일]
    DS --> MASK[masks/ 학습 Mask]
    DS --> MODEL[*.pt 모델 체크포인트]
    DS --> METRIC[metrics·overlay 결과]
    DATA --> CAMP[data/campuses.json 캠퍼스 목록]
    OUT --> GRAPH[graph_pilot/ 그래프 변환 결과]
```

## 3. 데이터 객체와 관계

```mermaid
erDiagram
    CAMPUS ||--o{ GRAPH_VERSION : has
    GRAPH_VERSION ||--o{ NAV_NODE : contains
    GRAPH_VERSION ||--o{ NAV_EDGE : contains
    NAV_NODE ||--o{ NAV_EDGE : source
    NAV_NODE ||--o{ NAV_EDGE : target
    IMAGE_TILE ||--|| LABEL_MASK : paired_with
    LABEL_MASK ||--o{ MODEL_RUN : evaluates
    MODEL_CHECKPOINT ||--o{ MODEL_RUN : used_by
    MODEL_RUN ||--o{ PREDICTION_MASK : produces
    PREDICTION_MASK ||--o{ GRAPH_VERSION : converted_to

    CAMPUS {
      string campus_id PK
      string name
      float center_lat
      float center_lon
    }
    GRAPH_VERSION {
      uuid version_id PK
      string campus_id FK
      string source
      string status
    }
    NAV_NODE {
      bigint node_id PK
      uuid version_id FK
      geometry point
    }
    NAV_EDGE {
      bigint edge_id PK
      uuid version_id FK
      bigint source_id FK
      bigint target_id FK
      geometry line
      float length_m
    }
    IMAGE_TILE {
      string tile_id PK
      string image_path
      int zoom
      int x
      int y
    }
    LABEL_MASK {
      string tile_id FK
      string mask_path
      string label_source
    }
    MODEL_CHECKPOINT {
      string model_id PK
      string architecture
      string checkpoint_path
    }
    MODEL_RUN {
      string run_id PK
      string model_id FK
      string dataset_version
      float dice
      float iou
    }
    PREDICTION_MASK {
      string run_id FK
      string tile_id FK
      string mask_path
    }
```

## 4. 학습 데이터 생성 흐름

```mermaid
flowchart LR
    OSM[OSM highway 보행로] --> V[LineString·GeoJSON]
    V --> W[Buffer 또는 고정 폭 Rasterize]
    AERIAL[Esri 항공영상 타일] --> PAIR[동일 tile_id 매칭]
    W --> PAIR
    PAIR --> QC[Overlay 사람 검수]
    QC --> SPLIT[Train / Validation 분리]
    SPLIT --> NORMALIZE[이미지 Resize·정규화]
    NORMALIZE --> BATCH[Image, Mask Batch]
```

현재 파일럿은 OSM 기반 pseudo-label과 고정 폭 rasterization을 사용하며, 실제 서비스 전에는 GIS 단위의 도로 폭 Buffer와 검수 라벨 재생성이 필요합니다.

## 5. 기본 U-Net 구조

```mermaid
flowchart LR
    X[입력 RGB 256×256×3] --> E1[Encoder 1\nConv 3→32]
    E1 --> P1[MaxPool]
    P1 --> E2[Encoder 2\nConv 32→64]
    E2 --> B[Bottleneck]
    B --> U[Upsample\nConvTranspose]
    U --> CAT[Skip Connection\nEncoder feature 결합]
    E1 -.-> CAT
    CAT --> D[Decoder\nConv 64→32]
    D --> Y[1×1 Conv]
    Y --> M[보행 확률 Mask\n256×256×1]
```

## 6. U-Net + MobileNetV3 전이학습 구조

```mermaid
flowchart LR
    X[입력 RGB 이미지] --> N[ImageNet 정규화]
    N --> MB[MobileNetV3-Small Encoder\nImageNet pretrained]
    MB --> F1[저수준 특징]
    MB --> F2[중간 특징]
    MB --> F3[고수준 특징]
    F1 --> D1[Decoder Block]
    F2 --> D2[Decoder Block]
    F3 --> D3[Bottleneck Decoder]
    D3 --> D2 --> D1 --> OUT[Segmentation Head]
    OUT --> MASK[보행 Mask]
    FT[Fine-tuning\nBCE + Dice Loss] -. 학습 .-> MB
```

두 모델은 같은 이미지·라벨·epoch·평가 기준으로 비교해야 하며, MobileNetV3 모델만 ImageNet 사전학습 가중치를 사용합니다.

## 7. 학습 루프

```mermaid
flowchart TD
    LOAD[Dataset Loader] --> AUG[Resize·Tensor 변환]
    AUG --> FORWARD[Forward Pass]
    FORWARD --> PRED[예측 확률]
    PRED --> LOSS[BCE Weighted + Dice Loss]
    MASK[정답 Mask] --> LOSS
    LOSS --> BACK[Backpropagation]
    BACK --> OPT[Optimizer Step]
    OPT --> NEXT{다음 Batch?}
    NEXT -- 예 --> FORWARD
    NEXT -- 아니오 --> EPOCH{다음 Epoch?}
    EPOCH -- 예 --> LOAD
    EPOCH -- 아니오 --> SAVE[Checkpoint·Metrics 저장]
```

## 8. 학습 전후 예측 비교

```mermaid
flowchart LR
    INPUT[동일 입력 타일] --> BEFORE[학습 전\nRandom Init U-Net]
    INPUT --> AFTER[학습 후\nTrained U-Net]
    BEFORE --> BM[Before Mask·Overlay]
    AFTER --> AM[After Mask·Overlay]
    GT[동일 Ground Truth Mask] --> SCORE1[Dice·IoU]
    BM --> SCORE1
    GT --> SCORE2[Dice·IoU]
    AM --> SCORE2
    SCORE1 --> COMP[동일 샘플 기준 비교]
    SCORE2 --> COMP
```

## 9. Mask에서 Graph로 변환

```mermaid
flowchart LR
    MASK[Segmentation Mask] --> CLEAN[Threshold·Noise 제거]
    CLEAN --> MORPH[Closing·Gap 연결]
    MORPH --> SKEL[Skeletonize 중심선화]
    SKEL --> DEG[픽셀 연결 차수 계산]
    DEG --> NODE[끝점·교차점·분기점 Node]
    SKEL --> TRACE[Node 사이 선 추적]
    TRACE --> EDGE[LineString Edge]
    NODE --> GEO[GeoJSON / 좌표 변환]
    EDGE --> GEO
    GEO --> DB[(PostGIS nav_nodes·nav_edges)]
```

현재 길찾기 서비스의 운영 그래프는 OSM 기반이며, 위 AI Mask→Graph 자동화는 다음 고도화 대상입니다.

## 10. PostGIS·API·웹 서비스 관계

```mermaid
sequenceDiagram
    participant U as 사용자
    participant W as Web Demo
    participant API as FastAPI
    participant DB as PostGIS
    U->>W: 캠퍼스 선택
    W->>API: GET /api/campuses
    API->>DB: 캠퍼스·published graph 조회
    DB-->>API: 캠퍼스 목록
    API-->>W: 캠퍼스 중심점·그래프 정보
    U->>W: 출발지·목적지 핀 선택
    W->>API: POST /api/routes
    API->>DB: nearest node + shortest path
    DB-->>API: Route geometry·거리·노드 수
    API-->>W: GeoJSON 경로
    W-->>U: 지도 위 경로 표시
```

## 11. 현재 구현과 목표 구현

```mermaid
flowchart TD
    NOW[현재 구현]
    NOW --> N1[OSM 기반 라벨·그래프]
    NOW --> N2[U-Net / MobileNetV3 파일럿 학습]
    NOW --> N3[PostGIS Route API]
    NOW --> N4[웹 라벨 검수·길찾기]
    FUTURE[고도화 목표]
    FUTURE --> F1[검수 GeoJSON 자동 재 rasterize]
    FUTURE --> F2[검수 라벨 기반 재학습]
    FUTURE --> F3[AI Mask 자동 Skeletonize·Graph 생성]
    FUTURE --> F4[Graph versioning·승인·배포]
    FUTURE --> F5[다중 캠퍼스 데이터 파이프라인]
```
