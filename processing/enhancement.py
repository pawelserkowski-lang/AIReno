
import cv2

def denoise(img, strength=10):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
