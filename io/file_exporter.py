
import cv2
import os
import datetime

def save_image(img, out_dir="output", base_name="restored", fmt="png"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{base_name}_{timestamp}.{fmt}")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path
