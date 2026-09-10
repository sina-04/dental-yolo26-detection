from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.prepare_dataset import parse_label, patient_key


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


if __name__ == "__main__":
    unittest.main()
