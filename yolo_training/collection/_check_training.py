"""One-off: check Roboflow training status and download weights."""
import os, json, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("D:/project/vehicle-sdi-system/.env"))
api_key = os.getenv("ROBOFLOW_API_KEY")

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
proj = rf.workspace().project("wheel-bbox-final")
v = proj.version(3)

# Print model info
print("Model:", v.model)
print()

# Try an inference to check the model is live
try:
    pred = v.model.predict(
        "d:/project/vehicle-sdi-system/yolo_training/dataset/images/val/000001.jpg",
        confidence=30,
    )
    print(f"Test inference: {len(pred.json()['predictions'])} detections")
    for p in pred.json()["predictions"]:
        print(f"  class={p['class']}, conf={p['confidence']:.2f}, bbox=({p['x']:.0f},{p['y']:.0f},{p['width']:.0f},{p['height']:.0f})")
except Exception as e:
    print(f"Inference failed: {e}")
