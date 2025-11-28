
import cv2
import numpy as np
import os

def load_image(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plik nie istnieje: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
