import os
import csv
import base64
import numpy as np
import cv2

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(LOG_DIR, "mispredictions.csv")

def save_canvas_image_and_meta(canvas_rgba, pred, conf, true_label=None):
    # save image (png)
    idx = len(os.listdir(LOG_DIR))
    fname = f"canvas_{idx}.png"
    p = os.path.join(LOG_DIR, fname)
    # convert rgba to bgr and save
    img_bgr = cv2.cvtColor(canvas_rgba.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    cv2.imwrite(p, img_bgr)
    # append metadata
    header = ["filename", "pred", "confidence", "true_label"]
    row = [fname, int(pred), float(conf), true_label if true_label is not None else ""]
    write_header = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
