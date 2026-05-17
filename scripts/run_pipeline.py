"""Legacy-compatible pipeline entrypoint for sdi-helper.

This script keeps the old run_pipeline ergonomics while delegating execution to
`python -m sdi_helper.interfaces.cli.run_scrape`.

Compatibility behavior:
- If LOCAL_DATASET_ROOT is unset and STANFORD_CARS_EXTRACTED_ROOT is set,
  LOCAL_DATASET_ROOT is derived from it.
- LOCAL_MAX_IMAGES maps to --max-results (single-query run).
"""

from __future__ import annotations

import os
import runpy
import sys


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def main() -> int:
    # Preserve old env semantics for local dataset root.
    if not os.getenv("LOCAL_DATASET_ROOT") and os.getenv("STANFORD_CARS_EXTRACTED_ROOT"):
        os.environ["LOCAL_DATASET_ROOT"] = os.environ["STANFORD_CARS_EXTRACTED_ROOT"]

    max_images = _as_int(os.getenv("LOCAL_MAX_IMAGES"), default=0)

    # Delegate to the maintained CLI pipeline implementation.
    argv = ["run_scrape.py", "--max-queries", "1", "--verbose"]
    if max_images > 0:
        argv.extend(["--max-results", str(max_images)])

    sys.argv = argv
    runpy.run_module("sdi_helper.interfaces.cli.run_scrape", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
