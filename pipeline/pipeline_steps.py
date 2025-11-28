
from io.file_exporter import save_image
from processing.segmentation import detect_faces
from processing.enhancement import denoise
from processing.color_ops import white_balance
from face.face_restore import soft_face_restore

def step_detect_faces(img, ctx):
    ctx["faces"] = detect_faces(img)
    return img

def step_basic_cleanup(img, ctx):
    return denoise(img, strength=12)

def step_face_restore(img, ctx):
    faces = ctx.get("faces", [])
    if len(faces) == 0:
        return img
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    restored = soft_face_restore(face)
    img[y:y+h, x:x+w] = restored
    return img

def step_color_fix(img, ctx):
    return white_balance(img)

def step_export(img, ctx):
    return save_image(img, base_name="restored")
