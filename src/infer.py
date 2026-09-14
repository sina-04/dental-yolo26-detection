from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common import IMAGE_EXTENSIONS
from src.inference_utils import apply_class_thresholds, merge_detections, tile_windows

ROOT = Path(__file__).resolve().parents[1]


def resolve_confidence(explicit: float | None, metrics_path: Path) -> float:
    if explicit is not None:
        if not 0.0 <= explicit <= 1.0:
            raise ValueError("--conf must be between 0 and 1")
        return explicit
    if not metrics_path.exists():
        raise FileNotFoundError(
            "No confidence threshold was supplied and final_metrics.json is unavailable. "
            "Pass --conf or complete the test phase."
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics["validation_selected_threshold"])


def resolve_thresholds(explicit: float | None, metrics_path: Path) -> tuple[float | dict[int, float], dict[str, Any]]:
    if explicit is not None:
        return resolve_confidence(explicit, metrics_path), {}
    if not metrics_path.exists():
        return resolve_confidence(None, metrics_path), {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    configured = metrics.get("validation_selected_thresholds")
    if configured:
        thresholds = {
            int(class_id): float(value["threshold"] if isinstance(value, dict) else value)
            for class_id, value in configured.items()
        }
        return thresholds, dict(metrics.get("tile_inference", {}))
    return float(metrics["validation_selected_threshold"]), dict(metrics.get("tile_inference", {}))


def class_name(names: Any, class_id: int) -> str:
    return str(names[class_id] if not isinstance(names, dict) else names.get(class_id, class_id))


def _local_sources(source: str) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def _boxes(result: Any, x_offset: int = 0, source_view: str = "full") -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls.item())
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        xyxy[0] += x_offset
        xyxy[2] += x_offset
        detections.append({
            "class_id": class_id,
            "class_name": class_name(result.names, class_id),
            "confidence": float(box.conf.item()),
            "xyxy": xyxy,
            "source_view": source_view,
        })
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected panoramic dental detector on images.")
    parser.add_argument("source", help="Image, directory, glob, URL, or camera source accepted by Ultralytics")
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "inference_output")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports" / "final_metrics.json")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("auto", "full", "hybrid_tiles"), default="auto")
    parser.add_argument("--tile-count", type=int, default=3)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--tile-aspect-ratio", type=float, default=1.4)
    parser.add_argument("--merge-iou", type=float, default=0.55)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    thresholds, tile_configuration = resolve_thresholds(args.conf, args.metrics)
    confidence = min(thresholds.values()) if isinstance(thresholds, dict) else thresholds
    args.output.mkdir(parents=True, exist_ok=True)

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    device = None if args.device == "auto" else args.device
    local_sources = _local_sources(args.source)
    mode = args.mode
    if mode == "auto":
        mode = "hybrid_tiles" if tile_configuration.get("enabled") and local_sources else "full"
    if mode == "hybrid_tiles" and not local_sources:
        raise ValueError("Hybrid tile inference requires a local image or directory source")
    tile_count = int(tile_configuration.get("tiles", args.tile_count))
    tile_overlap = float(tile_configuration.get("overlap", args.tile_overlap))
    aspect_trigger = float(tile_configuration.get("aspect_ratio_trigger", args.tile_aspect_ratio))
    merge_iou = float(tile_configuration.get("merge_iou", args.merge_iou))
    results = model.predict(source=args.source, conf=confidence, imgsz=args.imgsz, device=device, verbose=False) if mode == "full" else []
    summary: list[dict[str, object]] = []
    work_items = list(enumerate(results)) if mode == "full" else list(enumerate(local_sources))
    for index, item in work_items:
        if mode == "full":
            result = item
            source_path = Path(str(result.path))
            detections = _boxes(result)
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        else:
            source_path = item
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            full_result = model.predict(image, conf=confidence, imgsz=args.imgsz, device=device, verbose=False)[0]
            detections = _boxes(full_result)
            height, width = image.shape[:2]
            if width / max(height, 1) > aspect_trigger:
                for tile_index, (x1, y1, x2, y2) in enumerate(
                    tile_windows(width, height, tile_count, tile_overlap), start=1
                ):
                    tile_result = model.predict(
                        image[y1:y2, x1:x2], conf=confidence, imgsz=args.imgsz, device=device, verbose=False
                    )[0]
                    detections.extend(_boxes(tile_result, x_offset=x1, source_view=f"tile_{tile_index}"))
            detections = merge_detections(detections, merge_iou)
        detections = apply_class_thresholds(detections, thresholds)
        stem = source_path.stem or f"prediction_{index:04d}"
        image_output = args.output / f"{stem}_annotated.jpg"
        json_output = args.output / f"{stem}_detections.json"
        if image_output.exists() or json_output.exists():
            image_output = args.output / f"{stem}_{index:04d}_annotated.jpg"
            json_output = args.output / f"{stem}_{index:04d}_detections.json"
        if image is None:
            continue
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["xyxy"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (40, 40, 240), 2)
            cv2.putText(
                image,
                f"{detection['class_name']} {detection['confidence']:.2f}",
                (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 240), 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(image_output), image)
        payload = {
            "source": str(source_path),
            "weights": str(args.weights.resolve()),
            "confidence_thresholds": thresholds,
            "inference_mode": mode,
            "detections": detections,
            "annotated_image": str(image_output.resolve()),
        }
        json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary.append({"source": str(source_path), "detections": len(detections), "json": str(json_output)})
    print(json.dumps({"confidence_thresholds": thresholds, "inference_mode": mode, "results": summary}, indent=2))


if __name__ == "__main__":
    main()
