"""Audit all images in side_view_dataset/images/all/ against strict prerequisites."""
from pathlib import Path
from PIL import Image
import numpy as np

base = Path("yolo_training/side_view_dataset/images/all")
imgs = sorted(f for f in base.iterdir() if f.is_file())

print(f"Total images in all/: {len(imgs)}")
print()

stats = []
for p in imgs:
    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        aspect = w / h if h > 0 else 0
        arr = np.array(img)

        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        overall_std = arr.std()

        # Background brightness: top 15%, left 10%, right 10%
        top_strip = arr[: int(h * 0.15), :, :]
        left_strip = arr[:, : int(w * 0.1), :]
        right_strip = arr[:, int(w * 0.9) :, :]
        bg_brightness = np.mean(
            [top_strip.mean(), left_strip.mean(), right_strip.mean()]
        )

        # Unique colors at 128x128
        small = img.resize((128, 128), Image.LANCZOS)
        small_arr = np.array(small)
        flat = small_arr.reshape(-1, 3)
        unique_colors = len(set(map(tuple, flat.tolist())))

        # Bottom darkness (shadow)
        bottom_strip = arr[int(h * 0.85) :, :, :]
        dark_ratio = float((bottom_strip.mean(axis=2) < 60).mean())

        flags = []

        # Prerequisite 1: orientation — aspect ratio proxy
        if aspect < 1.15:
            flags.append(f"PORTRAIT_OR_SQUARE:aspect={aspect:.2f}")
        elif aspect > 3.0:
            flags.append(f"PANORAMIC:aspect={aspect:.2f}")

        # Prerequisite 3: illustration / render detection
        if unique_colors < 5000:
            flags.append(f"LOW_COLORS:{unique_colors}")
        if overall_std < 30:
            flags.append(f"LOW_VARIANCE:{overall_std:.1f}")

        # Background check — street scene indicator
        if bg_brightness < 140:
            flags.append(f"DARK_BG:{bg_brightness:.0f}")

        # Shadow under vehicle
        if dark_ratio > 0.25:
            flags.append(f"HEAVY_SHADOW:{dark_ratio:.2f}")

        stats.append(
            {
                "name": p.name,
                "w": w,
                "h": h,
                "aspect": aspect,
                "bg_brightness": bg_brightness,
                "unique_colors": unique_colors,
                "overall_std": overall_std,
                "dark_ratio": dark_ratio,
                "flags": flags,
            }
        )

    except Exception as e:
        print(f"  ERROR: {p.name}: {e}")
        stats.append({"name": p.name, "flags": ["CORRUPT"]})

# Print flagged images
print("=== FLAGGED IMAGES (potential side_view_invalid) ===")
for s in stats:
    if s["flags"]:
        name = s["name"]
        dims = f'{s.get("w", "?")}x{s.get("h", "?")}'
        flag_str = " | ".join(s["flags"])
        print(f"  {name:<55s} {dims:<12s} {flag_str}")

flagged = sum(1 for s in stats if s["flags"])
clean = len(stats) - flagged
print()
print(f"Total: {len(stats)}")
print(f"Flagged: {flagged}")
print(f"Clean: {clean}")

# Aspect ratio summary
aspects = [s["aspect"] for s in stats if "aspect" in s]
print(f"\nAspect range: {min(aspects):.2f} — {max(aspects):.2f}")
below_1_3 = [s for s in stats if s.get("aspect", 99) < 1.3]
print(f"Narrow (<1.30): {len(below_1_3)}")
for s in below_1_3:
    print(f'  {s["name"]:<55s} {s["w"]}x{s["h"]}  aspect={s["aspect"]:.2f}')

# Color uniqueness distribution
uc = sorted(s.get("unique_colors", 0) for s in stats)
print(f"\nUnique colors (128px): min={uc[0]}, p10={uc[len(uc)//10]}, median={uc[len(uc)//2]}, max={uc[-1]}")

# BG brightness distribution
bgs = sorted(s.get("bg_brightness", 0) for s in stats)
print(f"BG brightness: min={bgs[0]:.0f}, p10={bgs[len(bgs)//10]:.0f}, median={bgs[len(bgs)//2]:.0f}, max={bgs[-1]:.0f}")
