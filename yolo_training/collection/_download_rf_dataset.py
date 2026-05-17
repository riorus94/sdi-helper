"""Download Roboflow v3 augmented dataset and train locally with ultralytics."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("D:/project/vehicle-sdi-system/.env"))

from roboflow import Roboflow

api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)
proj = rf.workspace().project("wheel-bbox-final")
v = proj.version(3)

# Download in YOLOv8 format to a local directory
dest = str(Path("D:/project/vehicle-sdi-system/yolo_training/dataset/roboflow_v3"))
print(f"Downloading augmented dataset to {dest}...")
ds = v.download("yolov8", location=dest)
print(f"Done. Dataset location: {ds.location}")

# List contents
import glob
for d in ["train", "valid", "test"]:
    imgs = glob.glob(os.path.join(dest, d, "images", "*"))
    lbls = glob.glob(os.path.join(dest, d, "labels", "*"))
    print(f"  {d}: {len(imgs)} images, {len(lbls)} labels")
