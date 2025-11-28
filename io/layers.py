
import numpy as np

def add_alpha(img):
    if img.shape[2] == 4:
        return img
    h, w, _ = img.shape
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([img, alpha], axis=2)

def blend(fg, bg, alpha):
    alpha_norm = alpha.astype(float) / 255.0
    return (fg * alpha_norm + bg * (1 - alpha_norm)).astype(np.uint8)
