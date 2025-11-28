
from io.file_loader import load_image
from pipeline import pipeline_steps as steps

DEFAULT_PIPELINE = [
    steps.step_detect_faces,
    steps.step_basic_cleanup,
    steps.step_face_restore,
    steps.step_color_fix,
    steps.step_export,
]

def run_pipeline(path: str, custom_pipeline=None):
    pipeline = custom_pipeline or DEFAULT_PIPELINE
    ctx = {}

    img = load_image(path)

    for step in pipeline:
        img = step(img, ctx)

    return img, ctx
