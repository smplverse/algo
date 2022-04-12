import numpy as np

from tqdm import tqdm
from src.onnx_model import OnnxModel
from src.detector import Detector
from src.data import get_smpls


def make_smpls_embeddings():
    paths, smpls = get_smpls("data/smpls")
    _zip = zip(paths, smpls)
    vgg = OnnxModel()
    fd = Detector()
    embeddings = {}
    for _ in tqdm(range(len(smpls))):
        path, smpl = _zip.__next__()
        smpl_face = fd.detect_face(smpl)
        if smpl_face is None:
            embeddings[path] = None
            continue
        embedding = vgg(smpl)
        embeddings[path] = embedding
    np.save("data/embeddings_vggface2.npy", embeddings)
