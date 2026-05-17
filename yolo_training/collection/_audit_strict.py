"""Strict prerequisite audit — only definitive violation signals."""
from pathlib import Path
from PIL import Image
import numpy as np

base = Path("yolo_training/side_view_dataset/images/all")

strict_flags = []
all_stats = []

for p in sorted(base.iterdir()):
    if not p.is_file():
        continue
    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        aspect = w / h if h > 0 else 0
        arr = np.array(img)

        # Unique colors at 128px
        small = img.resize((128, 128), Image.LANCZOS)
        flat = np.array(small).reshape(-1, 3)
        uc128 = len(set(map(tuple, flat.tolist())))

        overall_std = float(arr.std())

        # BG brightness
        top_strip = arr[: int(h * 0.15), :, :]
        left_strip = arr[:, : int(w * 0.1), :]
        right_strip = arr[:, int(w * 0.9) :, :]
        bg = float(
            np.mean([top_strip.mean(), left_strip.mean(), right_strip.mean()])
        )

        # Bottom darkness
        bottom = arr[int(h * 0.85) :, :, :]
        dark_ratio = float((bottom.mean(axis=2) < 60).mean())

        all_stats.append(
            {
                "name": p.name,
                "w": w,
                "h": h,
                "aspect": aspect,
                "uc128": uc128,
                "overall_std": overall_std,
                "bg": bg,
                "dark_ratio": dark_ratio,
            }
        )

        flags = []

        # 1. Illustration / vector / diagram — very few unique colors
        if uc128 < 1500:
            flags.append("ILLUSTRATION:uc128=%d" % uc128)

        # 2. Portrait or square — NOT a side view
        if aspect < 1.15:
            flags.append("PORTRAIT:aspect=%.2f" % aspect)

        # 3. Street scene / very dark background
        if bg < 100 and dark_ratio > 0.5:
            flags.append("STREET_DARK:bg=%.0f,shadow=%.2f" % (bg, dark_ratio))
        elif bg < 40:
            flags.append("VERY_DARK_BG:%.0f" % bg)

        # 4. Nearly zero variance — solid color / blank
        if overall_std < 15:
            flags.append("BLANK:std=%.1f" % overall_std)

        if flags:
            strict_flags.append((p.name, w, h, aspect, uc128, bg, dark_ratio, flags))

    except Exception as e:
        strict_flags.append((p.name, 0, 0, 0, 0, 0, 0, ["CORRUPT:" + str(e)]))

print("Strictly flagged: %d / %d" % (len(strict_flags), len(all_stats)))
print()
for name, w, h, aspect, uc, bg, dr, flags in strict_flags:
    flag_str = " | ".join(flags)
    print("  %-55s %dx%d  %s" % (name, w, h, flag_str))

# Distribution stats
print()
ucs = sorted(s["uc128"] for s in all_stats)
print("Unique colors (128px) distribution:")
print("  min=%d  p5=%d  p10=%d  p25=%d  median=%d  p75=%d  p90=%d  max=%d" % (
    ucs[0],
    ucs[int(len(ucs) * 0.05)],
    ucs[int(len(ucs) * 0.10)],
    ucs[int(len(ucs) * 0.25)],
    ucs[len(ucs) // 2],
    ucs[int(len(ucs) * 0.75)],
    ucs[int(len(ucs) * 0.90)],
    ucs[-1],
))

bgs = sorted(s["bg"] for s in all_stats)
print("BG brightness distribution:")
print("  min=%.0f  p10=%.0f  p25=%.0f  median=%.0f  p75=%.0f  max=%.0f" % (
    bgs[0],
    bgs[int(len(bgs) * 0.10)],
    bgs[int(len(bgs) * 0.25)],
    bgs[len(bgs) // 2],
    bgs[int(len(bgs) * 0.75)],
    bgs[-1],
))

aspects = sorted(s["aspect"] for s in all_stats)
print("Aspect ratio distribution:")
print("  min=%.2f  p10=%.2f  median=%.2f  p90=%.2f  max=%.2f" % (
    aspects[0],
    aspects[int(len(aspects) * 0.10)],
    aspects[len(aspects) // 2],
    aspects[int(len(aspects) * 0.90)],
    aspects[-1],
))
