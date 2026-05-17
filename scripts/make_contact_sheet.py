"""Generate an HTML contact sheet of all pending (unannotated, non-excepted) side-view images.

Usage:
    cd D:\project\sdi-helper
    .\.venv\Scripts\python.exe scripts\make_contact_sheet.py

Opens contact_sheet.html in the default browser when done.
Each image shows its stem below it.  Click an image to toggle it red (= mark as 3/4-view reject).
A "Copy rejected stems" button copies a YAML-ready list to the clipboard.
"""
from __future__ import annotations

import base64
import webbrowser
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
SOURCE_DIR = ROOT / "dataset_raw" / "images" / "train" / "side"
LABELME_JSON_DIR = ROOT / "yolo_training" / "side_view_dataset" / "labelme_json"
EXCEPTIONS_FILE = ROOT / "config" / "scrape_exceptions.yaml"
OUT_HTML = ROOT / "contact_sheet.html"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _annotated_stems() -> set[str]:
    if not LABELME_JSON_DIR.exists():
        return set()
    return {p.stem for p in LABELME_JSON_DIR.glob("*.json")}


def _exception_stems() -> set[str]:
    if not EXCEPTIONS_FILE.exists():
        return set()
    data = yaml.safe_load(EXCEPTIONS_FILE.read_text(encoding="utf-8")) or {}
    values = data.get("side_image_stems", [])
    return {str(v).strip() for v in values if str(v).strip()}


def _img_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return f"data:{mime};base64,{data}"


def main() -> None:
    annotated = _annotated_stems()
    exceptions = _exception_stems()

    all_images = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    pending = [p for p in all_images if p.stem not in annotated and p.stem not in exceptions]

    print(f"Total source images : {len(all_images)}")
    print(f"Already annotated   : {len(annotated)}")
    print(f"Excluded (exception): {len(exceptions)}")
    print(f"Pending (in sheet)  : {len(pending)}")

    cards = []
    for img in pending:
        src = _img_tag(img)
        cards.append(
            f'<div class="card" data-stem="{img.stem}" onclick="toggle(this)">'
            f'<img src="{src}" title="{img.name}">'
            f'<div class="stem">{img.stem[:16]}…</div>'
            f"</div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Side-view contact sheet — {len(pending)} pending</title>
<style>
  body {{ font-family: sans-serif; background:#1a1a1a; color:#eee; margin:0; padding:12px; }}
  h1 {{ font-size:1rem; margin-bottom:8px; }}
  #toolbar {{ margin-bottom:10px; display:flex; gap:8px; align-items:center; }}
  button {{ padding:6px 14px; cursor:pointer; border-radius:4px; border:none; font-size:.85rem; }}
  #copy-btn {{ background:#e74c3c; color:#fff; }}
  #clear-btn {{ background:#555; color:#fff; }}
  #count {{ font-size:.85rem; color:#aaa; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .card {{ width:180px; cursor:pointer; border:3px solid transparent; border-radius:6px;
           background:#2a2a2a; padding:4px; box-sizing:border-box; }}
  .card.reject {{ border-color:#e74c3c; background:#3a1a1a; }}
  .card img {{ width:100%; height:110px; object-fit:cover; border-radius:3px; display:block; }}
  .stem {{ font-size:.6rem; color:#aaa; margin-top:4px; word-break:break-all; }}
  pre {{ background:#111; padding:10px; border-radius:4px; font-size:.75rem;
         max-height:200px; overflow:auto; display:none; margin-top:10px; }}
</style>
</head>
<body>
<h1>Pending side-view images ({len(pending)}) — click to mark as 3/4-view reject</h1>
<div id="toolbar">
  <button id="copy-btn" onclick="copyStems()">Copy rejected stems (YAML)</button>
  <button id="clear-btn" onclick="clearAll()">Clear selection</button>
  <span id="count">0 selected</span>
</div>
<div class="grid">{''.join(cards)}</div>
<pre id="yaml-out"></pre>
<script>
function toggle(el) {{
  el.classList.toggle('reject');
  updateCount();
}}
function clearAll() {{
  document.querySelectorAll('.card.reject').forEach(el => el.classList.remove('reject'));
  updateCount();
}}
function updateCount() {{
  const n = document.querySelectorAll('.card.reject').length;
  document.getElementById('count').textContent = n + ' selected';
}}
function copyStems() {{
  const stems = [...document.querySelectorAll('.card.reject')].map(el => '  - ' + el.dataset.stem);
  if (!stems.length) {{ alert('Nothing selected.'); return; }}
  const yaml = stems.join('\\n');
  const pre = document.getElementById('yaml-out');
  pre.textContent = yaml;
  pre.style.display = 'block';
  navigator.clipboard.writeText(yaml).then(() => alert('Copied ' + stems.length + ' stems to clipboard.'));
}}
</script>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nWrote: {OUT_HTML}")
    webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
