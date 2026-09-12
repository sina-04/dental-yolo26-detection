from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.colab_workflow import audit_is_approved, build_parser, download_dataset, prepared_dataset_is_valid
from src.common import stable_fingerprint
from src.infer import resolve_confidence
from src.prepare_dataset import (
    Box,
    Record,
    assign_splits,
    augmentation_copy_plan,
    build_augmentation,
    find_near_duplicate_pairs,
    merge_near_duplicate_groups,
    parse_label,
    patient_key,
    resolve_dataset_root,
    write_training_views,
)
from src.train_evaluate import require_audit_approval, support_tier


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
        self.assertEqual("colab_t4.yaml", args.profile.name)
        self.assertIn("panoramic31-yolo26s-t4-v1", str(args.results_root))
        self.assertEqual("prepare", args.stage)
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
            (dataset / "runtime").mkdir()
            (dataset / "runtime" / "base.yaml").write_text("names: []\n", encoding="utf-8")
            reports.mkdir()
            (reports / "dataset_verification.json").write_text('{"status":"pass"}', encoding="utf-8")
            (dataset / "fingerprint.json").write_text('{"fingerprint":"abc"}', encoding="utf-8")
            self.assertFalse(prepared_dataset_is_valid(dataset, reports))
            (dataset / ".colab_prepared.json").write_text('{"dataset_fingerprint":"abc"}', encoding="utf-8")
            self.assertTrue(prepared_dataset_is_valid(dataset, reports))
            self.assertFalse(prepared_dataset_is_valid(dataset, reports, {"minority_target_instances": 128}))

    def test_dataset_fingerprint_is_order_stable_and_sensitive(self) -> None:
        self.assertEqual(stable_fingerprint({"a": 1, "b": 2}), stable_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(stable_fingerprint({"a": 1}), stable_fingerprint({"a": 2}))

    def test_near_duplicates_are_transitively_grouped(self) -> None:
        records = [
            Record("panoramic", "train", Path(f"{index}.jpg"), None, f"p{index}", phash=value, sha256=str(index))
            for index, value in enumerate(("0000000000000000", "0000000000000001", "0000000000000003"))
        ]
        pairs = find_near_duplicate_pairs(records, max_distance=1)
        merge_near_duplicate_groups(records, pairs)
        self.assertEqual(2, len(pairs))
        self.assertEqual(1, len({record.patient_group for record in records}))

    def test_ultra_rare_class_is_forced_to_training(self) -> None:
        records = [
            Record("panoramic", "train", Path("rare.jpg"), None, "rare_group", [Box(0, 0.5, 0.5, 0.1, 0.1)]),
            Record("panoramic", "train", Path("common1.jpg"), None, "common_1", [Box(1, 0.5, 0.5, 0.1, 0.1)]),
            Record("panoramic", "train", Path("common2.jpg"), None, "common_2", [Box(1, 0.5, 0.5, 0.1, 0.1)]),
            Record("panoramic", "train", Path("common3.jpg"), None, "common_3", [Box(1, 0.5, 0.5, 0.1, 0.1)]),
        ]
        metadata = assign_splits(records, class_count=2)
        self.assertEqual("train", records[0].split)
        self.assertIn(0, metadata["forced_train_classes"])

    def test_training_views_isolate_augmented_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            train = dataset / "images" / "train"
            train.mkdir(parents=True)
            (train / "base.jpg").write_bytes(b"base")
            (train / "base_aug01.jpg").write_bytes(b"aug")
            counts = write_training_views(dataset, [f"class_{index}" for index in range(31)])
            base = (dataset / "runtime" / "base_train.txt").read_text(encoding="utf-8")
            augmented = (dataset / "runtime" / "augmented_train.txt").read_text(encoding="utf-8")
        self.assertEqual({"base": 1, "augmented": 2}, counts)
        self.assertNotIn("_aug", base)
        self.assertIn("_aug", augmented)

    def test_audit_approval_must_match_dataset_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, reports, results = root / "dataset", root / "reports", root / "results"
            dataset.mkdir()
            reports.mkdir()
            (results / "reports").mkdir(parents=True)
            (dataset / "fingerprint.json").write_text('{"fingerprint":"current"}', encoding="utf-8")
            approval = {"status": "approved", "dataset_fingerprint": "stale"}
            (reports / "manual_audit_approval.json").write_text(json.dumps(approval), encoding="utf-8")
            self.assertFalse(audit_is_approved(dataset, reports, results))
            approval["dataset_fingerprint"] = "current"
            (reports / "manual_audit_approval.json").write_text(json.dumps(approval), encoding="utf-8")
            self.assertTrue(audit_is_approved(dataset, reports, results))
            self.assertEqual("approved", require_audit_approval(reports / "manual_audit_approval.json", "current")["status"])
            with self.assertRaises(RuntimeError):
                require_audit_approval(reports / "manual_audit_approval.json", "different")

    def test_inference_threshold_uses_validation_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "final_metrics.json"
            metrics.write_text('{"validation_selected_threshold":0.35}', encoding="utf-8")
            self.assertEqual(0.35, resolve_confidence(None, metrics))
            self.assertEqual(0.5, resolve_confidence(0.5, metrics))

    def test_support_tiers_are_explicit(self) -> None:
        self.assertEqual("N/E", support_tier(0))
        self.assertEqual("very_low", support_tier(3))
        self.assertEqual("limited", support_tier(20))
        self.assertEqual("supported", support_tier(50))

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
