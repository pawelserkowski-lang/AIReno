
import cv2

def soft_face_restore(face_region):
    restored = cv2.bilateralFilter(face_region, 5, 50, 50)
    return restored
