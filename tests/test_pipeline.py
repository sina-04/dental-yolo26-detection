from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.colab_workflow import build_parser
from src.prepare_dataset import build_augmentation, parse_label, patient_key, resolve_dataset_root


class PipelineTests(unittest.TestCase):
    def test_detection_line_is_preserved_with_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("2 0.5 0.4 0.2 0.1\n", encoding="utf-8")
            boxes, issues, polygons = parse_label(label, class_offset=7, class_count=31)
        self.assertEqual(9, boxes[0].class_id)
        self.assertEqual((0.5, 0.4, 0.2, 0.1), (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h))
        self.assertEqual([], issues)
        self.assertEqual(0, polygons)

    def test_segmentation_polygon_becomes_bounding_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("1 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7\n", encoding="utf-8")
            boxes, issues, polygons = parse_label(label, class_offset=7, class_count=31)
        self.assertEqual(8, boxes[0].class_id)
        self.assertAlmostEqual(0.4, boxes[0].x)
        self.assertAlmostEqual(0.5, boxes[0].y)
        self.assertAlmostEqual(0.4, boxes[0].w)
        self.assertAlmostEqual(0.4, boxes[0].h)
        self.assertEqual([], issues)
        self.assertEqual(1, polygons)

    def test_patient_group_hides_cleartext_name_and_groups_roboflow_variants(self) -> None:
        first = patient_key("disease", "000dc27f-PATIENT_NAME_2020-07-12_jpg.rf.aaaaaaaa")
        second = patient_key("disease", "000dc27f-PATIENT_NAME_2020-07-12_jpg.rf.bbbbbbbb")
        self.assertEqual(first, second)
        self.assertNotIn("patient", first)
        self.assertNotIn("name", first)

    def test_dataset_root_is_resolved_from_nested_colab_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "download" / "Dental Dataset"
            nested.mkdir(parents=True)
            nested.joinpath("data.yaml").write_text(
                "names:\n" + "".join(f"  {index}: class_{index}\n" for index in range(7)),
                encoding="utf-8",
            )
            resolved = resolve_dataset_root(None, [Path(directory)], 7, "test")
        self.assertEqual(nested.resolve(), resolved)

    def test_colab_profile_defaults_to_full_data_yolo26s(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual("yolo26s.pt", args.model)
        self.assertIsNone(getattr(args, "max_train_images", None))
        self.assertEqual(640, args.imgsz)

    def test_augmentation_policy_builds_with_installed_backend(self) -> None:
        self.assertIsNotNone(build_augmentation(seed=42))


if __name__ == "__main__":
    unittest.main()
