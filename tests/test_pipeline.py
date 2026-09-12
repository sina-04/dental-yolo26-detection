from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.colab_workflow import build_parser, download_dataset, prepared_dataset_is_valid
from src.prepare_dataset import (
    Box,
    Record,
    augmentation_copy_plan,
    build_augmentation,
    parse_label,
    patient_key,
    resolve_dataset_root,
)


class PipelineTests(unittest.TestCase):
    def test_detection_line_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("2 0.5 0.4 0.2 0.1\n", encoding="utf-8")
            boxes, issues, polygons = parse_label(label, class_count=31)
        self.assertEqual(2, boxes[0].class_id)
        self.assertEqual((0.5, 0.4, 0.2, 0.1), (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h))
        self.assertEqual([], issues)
        self.assertEqual(0, polygons)

    def test_segmentation_polygon_becomes_bounding_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("1 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7\n", encoding="utf-8")
            boxes, issues, polygons = parse_label(label, class_count=31)
        self.assertEqual(1, boxes[0].class_id)
        self.assertAlmostEqual(0.4, boxes[0].x)
        self.assertAlmostEqual(0.5, boxes[0].y)
        self.assertAlmostEqual(0.4, boxes[0].w)
        self.assertAlmostEqual(0.4, boxes[0].h)
        self.assertEqual([], issues)
        self.assertEqual(1, polygons)

    def test_patient_group_hides_cleartext_name_and_groups_roboflow_variants(self) -> None:
        first = patient_key("000dc27f-PATIENT_NAME_2020-07-12_jpg.rf.aaaaaaaa")
        second = patient_key("000dc27f-PATIENT_NAME_2020-07-12_jpg.rf.bbbbbbbb")
        self.assertEqual(first, second)
        self.assertNotIn("patient", first)
        self.assertNotIn("name", first)

    def test_dataset_root_is_resolved_from_nested_colab_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "download" / "Dental X-Ray Panoramic Dataset"
            nested.mkdir(parents=True)
            nested.joinpath("data.yaml").write_text(
                "names:\n" + "".join(f"  {index}: class_{index}\n" for index in range(31)),
                encoding="utf-8",
            )
            resolved = resolve_dataset_root(None, [Path(directory)], 31, "test")
        self.assertEqual(nested.resolve(), resolved)

    def test_colab_profile_defaults_to_full_data_yolo26s(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual("yolo26s.pt", args.model)
        self.assertIsNone(getattr(args, "max_train_images", None))
        self.assertEqual(640, args.imgsz)
        self.assertEqual(32, args.batch)
        self.assertEqual(100, args.tuned_epochs)
        self.assertEqual(20, args.patience)
        self.assertEqual(128, args.minority_target_instances)

    def test_class_aware_augmentation_caps_rare_patient_reuse(self) -> None:
        rare = Record("panoramic", "train", Path("rare.jpg"), None, "p1", [Box(0, 0.5, 0.5, 0.1, 0.1)], output_id="rare", split="train")
        common = Record(
            "panoramic",
            "train",
            Path("common.jpg"),
            None,
            "p2",
            [Box(1, 0.5, 0.5, 0.1, 0.1) for _ in range(20)],
            output_id="common",
            split="train",
        )
        plan = augmentation_copy_plan([rare, common], 0.0, minority_target_instances=8, max_augmentations_per_image=3, seed=42)
        self.assertEqual(3, plan["rare"])
        self.assertEqual(0, plan["common"])

    def test_augmentation_policy_builds_with_installed_backend(self) -> None:
        self.assertIsNotNone(build_augmentation(seed=42))

    def test_colab_prepared_dataset_requires_session_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            reports = root / "reports"
            (dataset / "images" / "train").mkdir(parents=True)
            reports.mkdir()
            (reports / "dataset_verification.json").write_text('{"status":"pass"}', encoding="utf-8")
            self.assertFalse(prepared_dataset_is_valid(dataset, reports))
            (dataset / ".colab_prepared.json").write_text("{}", encoding="utf-8")
            self.assertTrue(prepared_dataset_is_valid(dataset, reports))
            self.assertFalse(prepared_dataset_is_valid(dataset, reports, {"minority_target_instances": 128}))

    def test_colab_download_marker_does_not_conflict_with_kagglehub_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "download"
            (destination / ".complete").mkdir(parents=True)
            destination.joinpath("data.yaml").write_text("names: []\n", encoding="utf-8")
            destination.joinpath("image.jpg").write_bytes(b"image")

            resolved = download_dataset("owner/dataset/versions/1", destination)

            self.assertEqual(destination, resolved)
            self.assertTrue((destination / ".complete").is_dir())
            self.assertEqual("owner/dataset/versions/1\n", (destination / ".codex_complete").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
