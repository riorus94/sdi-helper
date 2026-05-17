"""
Script to split side-view validity dataset into YOLO classification format.
- Reads images from yolo_training/side_view_scrape/images/raw/{valid,invalid}/
- Splits into train/val (80/20) with class subfolders
- Output: yolo_training/side_view_scrape/images/{train,val}/{side_view_valid,side_view_invalid}/
"""
import os
from pathlib import Path
import random
import shutil

RAW_DIR = Path('yolo_training/side_view_scrape/images/raw')
OUT_DIR = Path('yolo_training/side_view_scrape/images')
CLASSES = {'valid': 'side_view_valid', 'invalid': 'side_view_invalid'}
SPLITS = {'train': 0.8, 'val': 0.2}
SEED = 42

random.seed(SEED)

def gather_images(class_key):
    class_dir = RAW_DIR / class_key
    # Gather all images in all subfolders
    return [p for p in class_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

def split_and_copy():
    for class_key, class_name in CLASSES.items():
        images = gather_images(class_key)
        random.shuffle(images)
        n_train = int(len(images) * SPLITS['train'])
        split_map = {'train': images[:n_train], 'val': images[n_train:]}
        for split, split_imgs in split_map.items():
            out_dir = OUT_DIR / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                dest = out_dir / img.name
                shutil.copy2(img, dest)
    print('Dataset split complete.')

if __name__ == '__main__':
    split_and_copy()
    print('Output structure:')
    for split in SPLITS:
        for class_name in CLASSES.values():
            d = OUT_DIR / split / class_name
            print(f'  {d}: {len(list(d.glob("*")))} images')
