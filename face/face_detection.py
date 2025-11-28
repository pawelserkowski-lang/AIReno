
from processing.segmentation import detect_faces

def get_main_face(img):
    faces = detect_faces(img)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])
