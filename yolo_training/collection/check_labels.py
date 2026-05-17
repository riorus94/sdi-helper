"""Scan dataset for images with suspicious labels (!=2 wheels, huge/tiny bboxes)."""
import pathlib

DATASET = pathlib.Path(__file__).parent.parent / "dataset"

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def _iter_images(directory: pathlib.Path):
    """Yield image paths for all supported extensions, sorted."""
    images = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(directory.glob(pattern))
    return sorted(images, key=lambda p: p.name)


def scan(split: str):
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    issues = []
    for img in _iter_images(img_dir):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            issues.append((img.name, "NO LABEL"))
            continue
        lines = [l.strip() for l in lbl.read_text().splitlines() if l.strip()]
        n = len(lines)
        if n != 2:
            issues.append((img.name, f"{n} wheels"))
            continue
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                w, h = float(parts[3]), float(parts[4])
                if w > 0.45 or h > 0.45:
                    issues.append((img.name, f"huge bbox w={w:.3f} h={h:.3f}"))
                    break
                if w < 0.02 or h < 0.02:
                    issues.append((img.name, f"tiny bbox w={w:.3f} h={h:.3f}"))
                    break
    return issues


def _count_images(directory: pathlib.Path) -> int:
    return len(_iter_images(directory))


print("=== TRAIN ===")
train_issues = scan("train")
for name, issue in train_issues:
    print(f"  {name}: {issue}")
print(f"  ({len(train_issues)} flagged / {_count_images(DATASET / 'images' / 'train')} total)")

print("\n=== VAL ===")
val_issues = scan("val")
for name, issue in val_issues:
    print(f"  {name}: {issue}")
print(f"  ({len(val_issues)} flagged / {_count_images(DATASET / 'images' / 'val')} total)")
