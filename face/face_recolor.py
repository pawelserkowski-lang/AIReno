
import cv2
import numpy as np

def match_skin_tone(face, reference_mean=None):
    if reference_mean is None:
        return face
    yuv = cv2.cvtColor(face, cv2.COLOR_RGB2YUV)
    mean_face = yuv[:,:,0].mean()
    factor = reference_mean / (mean_face + 1e-5)
    yuv[:,:,0] = np.clip(yuv[:,:,0] * factor, 0,255)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
