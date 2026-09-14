from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.common import write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record clinician completion of the six-class pathology audit.")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument(
        "--confirm-complete", action="store_true",
        help="Confirm that primary-class validation/test annotations and mined errors were reviewed.",
    )
    args = parser.parse_args()
    if not args.confirm_complete:
        raise ValueError("Pass --confirm-complete only after the required clinician review is finished")
    reports = args.reports_root.resolve()
    support = json.loads((reports / "pathology_support.json").read_text(encoding="utf-8"))
    audit = json.loads((reports / "dataset_audit.json").read_text(encoding="utf-8"))
    payload = {
        "status": "approved",
        "dataset_fingerprint": audit["dataset_fingerprint"],
        "pathology_view_fingerprint": support["fingerprint"],
        "reviewer": args.reviewer.strip(),
        "credentials": args.credentials.strip(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": "all primary-class validation/test annotations and model-mined errors",
        "notes": args.notes.strip(),
    }
    if not payload["reviewer"] or not payload["credentials"] or not payload["notes"]:
        raise ValueError("Reviewer, credentials, and notes must be non-empty")
    write_json(reports / "pathology_clinician_review.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
