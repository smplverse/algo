from tqdm import tqdm
from src.onnx_model import OnnxModel
from src.detector import Detector
from src.data import get_smpls
from src.utils import serialize


def make_smpls_embeddings(model_path: str):
    paths, smpls = get_smpls("data/smpls")
    _zip = zip(paths, smpls)
    vgg = OnnxModel(model_path)
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
    model_name = model_path.split("/")[-1].split(".")[0]
    serialize(embeddings), f"data/embeddings_{model_name}.p")
