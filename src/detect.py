from typing import Union
import numpy as np

from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'],
                   root='./models/insightface')
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_face(img: np.ndarray) -> Union[dict, None]:
    faces = app.get(img)
    if len(faces):
        return faces[0]
    return None


def crop_face(img: np.ndarray) -> Union[np.ndarray, None]:
    face = detect_face(img)
    if face and 'bbox' in face:
        [x1, y1, x2, y2] = [int(i) for i in face['bbox']]
        print([x1, x2, y1, y2])
        cropped = img[y1:y2, x1:x2]
        return cropped
    return None
