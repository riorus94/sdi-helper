"""One-shot migration helper from pipeline/ folder to new sdi_helper/ layout.

Copies dataset_raw + state files from the old `pipeline/` package layout
into the new layout, preserving uuids and quota counters.

Usage:
    python scripts/migrate_from_old_pipeline.py --from ./pipeline/dataset_raw --to ./dataset_raw
"""


def main() -> int:
    raise NotImplementedError("Optional - only if you have pre-existing dataset_raw to preserve")


if __name__ == "__main__":
    raise SystemExit(main())
