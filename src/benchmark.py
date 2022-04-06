import numpy as np
import time

from tqdm import tqdm
from deepface.commons.functions import detect_face
from src.data import get_smpls
from src.visualization import show_img_cv


def benchmark(headless: bool = True):
    # before implementing our own, lets try the deepface builtins
    _, smpls = get_smpls("./data/smpls/")
    detector_backends = [
        'mediapipe',
        'ssd',
        'dlib',
        'mtcnn',
        'retinaface',
        'opencv',
    ]
    for detector_backend in detector_backends:
        inference_times = []
        tic = time.time()
        success = 0
        for i in tqdm(range(len(smpls))):
            smpl = smpls[i]
            inference_tic = time.time()
            _, bbox = detect_face(
                smpl,
                detector_backend,
                enforce_detection=False,
            )
            inference_toc = time.time()
            inference_times.append(float(inference_toc - inference_tic))
            if not bbox[0] == 0 and not bbox[1] == 0:
                success += 1
            else:
                if not headless:
                    show_img_cv(smpl)
        toc = time.time()
        print("\n\n" + detector_backend)
        print("-" * len(detector_backend))
        print("success rate: %.3f" % float(success / len(smpls)))
        print("total time elapsed: %.2fs" % float(toc - tic))
        print("mean time per image: %.4fs" % np.mean(inference_times))
