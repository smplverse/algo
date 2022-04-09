import cv2
import numpy as np
import time

from hashlib import sha256
from src.detector import Detector
from src.distance import Distance
from src.utils import merge
from src.vgg_face2 import VGGFace2
from src.visualization import show_comparison_cv
from typing import Any


class Matcher:

    def __init__(
        self,
        model: Any = VGGFace2(),
        headless: bool = False,
        session: bool = False,
    ):
        self.headless = headless
        self.session = session
        self.model = model
        self.detector = Detector()
        if self.session:
            self.scores = []
            self.inference_times = []

    def summarize(self):
        self.best_score_idx = np.argmin(self.scores)
        best_match = self.scores[self.best_score_idx]
        landed = np.count_nonzero(np.array(self.scores) != 1)
        print("\ndetection rate: %.2f" % landed / len(self.scores))
        print("best match: %.2f" % best_match)
        if len(self.inference_times):
            print("average time per image: %.2fs" %
                  np.mean(self.inference_times))

    def write_results(self, face: np.ndarray, smpl: np.ndarray):
        if not self.best_score_idx:
            self.summarize()
        smpl = self.smpls[self.best_score_idx]
        merged = merge(face, smpl)
        fname = sha256().digest()
        cv2.imwrite(f"results/image/{fname}.png", merged)
        if not self.headless:
            show_comparison_cv(face, smpl, final=True)

    def match(self, img: np.ndarray, smpl: np.ndarray):
        inference_tic = time.time()

        face = self.detector.detect_face(img)
        smpl_face = self.detector.detect_face(smpl)

        face_repr = self.model(face)
        smpl_repr = self.model(smpl_face)
        assert face_repr.shape == smpl_repr.shape
        distance = Distance(smpl_repr, face_repr).cosine()

        inference_toc = time.time()
        if not self.headless:
            show_comparison_cv(face, smpl_face)
        if self.session:
            inference_time = float(inference_toc - inference_tic)
            self.inference_times.append(inference_time)
            self.scores.append(distance)
        return distance
