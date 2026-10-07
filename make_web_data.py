"""Convert the real NSU OSM extract into browser-friendly graph data."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "web_demo"
OUT.mkdir(exist_ok=True)

def main():
    root = ET.parse(ROOT / "nsu_map.osm").getroot()
    nodes = {int(n.attrib["id"]): [float(n.attrib["lon"]), float(n.attrib["lat"])] for n in root.findall("node")}
    walkable = {"footway", "path", "pedestrian", "steps", "service", "track", "cycleway"}
    edges, used = [], set()
    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if tags.get("highway") not in walkable:
            continue
        refs = [int(x.attrib["ref"]) for x in way.findall("nd") if int(x.attrib["ref"]) in nodes]
        for a, b in zip(refs, refs[1:]):
            if a == b: continue
            key = tuple(sorted((a, b)))
            if key in used: continue
            used.add(key); edges.append({"source": a, "target": b, "coords": [nodes[a], nodes[b]], "highway": tags.get("highway"), "weight": 1.0})
            used.update([key])
    used_nodes = sorted({x for e in edges for x in (e["source"], e["target"])})
    sample = "18_223653_102117"
    metrics_path = ROOT / "nsu_real_dataset/training_metrics.json"
    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"epochs": 10, "seconds": None, "seconds_per_epoch": None}
    data = {
        "sample_id": sample,
        "training_metrics": training_metrics,
        "center": [36.90855, 127.14278],
        "zoom": 16,
        "nodes": [{"id": n, "lon": nodes[n][0], "lat": nodes[n][1]} for n in used_nodes],
        "edges": edges,
        "training": {"epochs": list(range(1, 11)), "loss": [1.6171, 1.5636, 1.5739, 1.5858, 1.4467, 1.4372, 1.4476, 1.4379, 1.4360, 1.4296], "note": "U-Net pilot loss on OSM pseudo-labels"},
        "steps": [
            {"id": "input", "title": "1. 항공영상 입력", "image": f"../nsu_real_dataset/images/{sample}.jpg", "text": "남서울대학교 실제 항공영상 타일입니다.", "detail": "수집: 남서울대학교 캠퍼스 좌표 주변의 실제 영상 타일을 확보했습니다.\n저장: nsu_real_dataset/images/ 아래에 256×256 JPG 타일로 저장합니다.\n처리: 모든 타일은 같은 줌 레벨과 좌표 규칙을 사용하고, 파일명에 zoom/x/y 타일 좌표를 포함합니다.\n다음 단계로 전달: 항공영상 JPG와 동일 위치의 라벨 파일을 한 쌍으로 사용합니다."},
            {"id": "label", "title": "2. 학습 라벨", "image": "../nsu_real_dataset/label_overlay.png", "text": "동일 항공영상 위 노란색 영역이 학습 대상 보행로 라벨입니다.", "detail": "수집: OSM 공식 API에서 highway=footway, path, pedestrian, steps 등의 보행로를 받았습니다.\n라벨링: 보행로 LineString을 일정 폭으로 버퍼링해 보행 가능 영역 Polygon으로 만들고, 영상 픽셀에 rasterize합니다.\n저장: 원본 벡터는 nsu_real_dataset/osm_walkways.geojson, 학습 마스크는 nsu_real_dataset/masks/에 저장합니다.\n주의: 현재 라벨은 OSM 기반 pseudo-label이므로 항공영상과 비교해 사람이 검수해야 합니다."},
            {"id": "output", "title": "3. 모델 예측", "image": "../nsu_real_dataset/demo_overlay.png", "text": "동일 항공영상에 학습된 U-Net이 예측한 보행 가능 영역입니다.", "detail": "학습: uv run train_unet.py로 항공영상 JPG를 입력하고 OSM 기반 마스크를 정답으로 사용합니다.\n모델: 작은 U-Net encoder-decoder 구조를 사용하며, 보행 픽셀이 적은 문제를 위해 BCE 가중치와 Dice loss를 함께 사용합니다.\n추론: uv run demo.py가 학습된 unet_nsu_walkable.pt를 불러와 보행 확률을 계산합니다.\n출력: demo_predicted_mask.png와 항공영상 위에 표시한 demo_overlay.png를 생성합니다."},
        ]
    }
    (OUT / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"nodes={len(data['nodes'])} edges={len(edges)}")

if __name__ == "__main__": main()
