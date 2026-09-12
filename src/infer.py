from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def class_name(names: Any, class_id: int) -> str:
    return str(names[class_id] if not isinstance(names, dict) else names.get(class_id, class_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected panoramic dental detector on images.")
    parser.add_argument("source", help="Image, directory, glob, URL, or camera source accepted by Ultralytics")
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "inference_output")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports" / "final_metrics.json")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    confidence = resolve_confidence(args.conf, args.metrics)
    args.output.mkdir(parents=True, exist_ok=True)

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    device = None if args.device == "auto" else args.device
    results = model.predict(source=args.source, conf=confidence, imgsz=args.imgsz, device=device, verbose=False)
    summary: list[dict[str, object]] = []
    for index, result in enumerate(results):
        source_path = Path(str(result.path))
        stem = source_path.stem or f"prediction_{index:04d}"
        image_output = args.output / f"{stem}_annotated.jpg"
        json_output = args.output / f"{stem}_detections.json"
        if image_output.exists() or json_output.exists():
            image_output = args.output / f"{stem}_{index:04d}_annotated.jpg"
            json_output = args.output / f"{stem}_{index:04d}_detections.json"
        detections: list[dict[str, object]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name(result.names, class_id),
                    "confidence": float(box.conf.item()),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                })
        cv2.imwrite(str(image_output), result.plot())
        payload = {
            "source": str(source_path),
            "weights": str(args.weights.resolve()),
            "confidence_threshold": confidence,
            "detections": detections,
            "annotated_image": str(image_output.resolve()),
        }
        json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary.append({"source": str(source_path), "detections": len(detections), "json": str(json_output)})
    print(json.dumps({"confidence_threshold": confidence, "results": summary}, indent=2))


if __name__ == "__main__":
    main()
