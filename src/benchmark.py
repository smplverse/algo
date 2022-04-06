import sys
import signal
import numpy as np
import time
from tqdm import tqdm
from deepface.commons.functions import detect_face
from src.data import get_smpls


def benchmark():
    # before implementing our own, lets try the deepface builtins
    _, smpls = get_smpls("./data/smpls/")
    detector_backends = [
        'opencv',
        'ssd',
        'dlib',
        'mtcnn',
        'retinaface',
        'mediapipe',
    ]
    inference_times = []
    tic = time.time()
    success = 0
    for i in tqdm(range(len(smpls))):
        smpl = smpls[i]
        inference_tic = time.time()
        _, bbox = detect_face(
            smpl,
            detector_backends[0],
            enforce_detection=False,
        )
        inference_toc = time.time()
        inference_times.append(float(inference_toc - inference_tic))
        if not bbox[0] == 0 and not bbox[1] == 0:
            success += 1
    toc = time.time()
    print("success rate: %.3f" % float(success / len(smpls)))
    print("total time elapsed: %.2fs" % float(toc - tic))
    print("average time per image: %.4fs" % np.mean(inference_times))
