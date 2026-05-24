# Only for tif images (Satellite dataset) , convert to 3 channels if they are single channel (grayscale) images. This is necessary for some models that expect 3-channel input. The original single-channel images will be overwritten with the new 3-channel versions.
import cv2
import numpy as np
from pathlib import Path

def convert_to_3channel(root_dir):
    for tif_path in Path(root_dir).rglob("*.tif"):
        img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read: {tif_path}")
            continue
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img_3ch = cv2.cvtColor(img if img.ndim == 2 else img[:, :, 0],
                                   cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(tif_path), img_3ch)
            print(f"Converted: {tif_path}")

convert_to_3channel("folds_clahe/")