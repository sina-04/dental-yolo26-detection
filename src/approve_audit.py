from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.common import write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record completion of the manual annotation contact-sheet audit.")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True, help="Concise review findings and any classes needing correction.")
    args = parser.parse_args()
    reports = args.reports_root.resolve()
    status_path = reports / "manual_audit_status.json"
    if not status_path.exists():
        raise FileNotFoundError("Run dataset preparation before approving the annotation audit")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    fingerprint = status.get("dataset_fingerprint")
    if not fingerprint:
        raise ValueError("The audit status does not contain a dataset fingerprint")
    sheets = sorted((reports / "annotation_audit").glob("class_*.jpg"))
    if len(sheets) != 31:
        raise ValueError(f"Expected 31 class contact sheets, found {len(sheets)}")
    approval = {
        "status": "approved",
        "dataset_fingerprint": fingerprint,
        "reviewer": args.reviewer.strip(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contact_sheets_reviewed": len(sheets),
        "notes": args.notes.strip(),
    }
    if not approval["reviewer"] or not approval["notes"]:
        raise ValueError("Reviewer and notes must be non-empty")
    write_json(reports / "manual_audit_approval.json", approval)
    print(json.dumps(approval, indent=2))


if __name__ == "__main__":
    main()
