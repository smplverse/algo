from hashlib import sha256
import time
from typing import Any, List

import cv2
import numpy as np

from src.detector import Detector
from src.distance import Distance
from src.utils import deserialize, merge
from src.vgg_face2 import VGGFace2
from src.visualization import show_comparison_cv


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
        self.failed_detections = 0
        self.smpls_embeddings = deserialize("data/smpls_embeddings_vggface2.p")
        if self.session:
            self.scores = []
            self.inference_times = []

    def write_results(self, face: np.ndarray, smpls: List[np.ndarray]):
        best_score_idx = np.argmin(self.scores)
        best_match = self.scores[best_score_idx]
        landed = np.count_nonzero(np.array(self.scores) != 1)
        det_rate = landed / len(self.scores)
        smpl = smpls[best_score_idx]
        print("\ndetection rate: %.2f" % det_rate)
        print("best match: %.2f" % best_match)
        if len(self.inference_times):
            print("average time per image: %.2fs" %
                  np.mean(self.inference_times))
        merged = merge(face, smpl)
        fname = sha256(merged).digest()
        cv2.imwrite(f"results/image/{fname}.png", merged)
        if not self.headless:
            show_comparison_cv(face, smpl, final=True)

    def match_images(self, img: np.ndarray, smpl: np.ndarray):
        inference_tic = time.time()

        face = self.detector.detect_face(img)
        smpl_face = self.detector.detect_face(smpl)
        if smpl_face is None:
            self.failed_detections += 1
            return None
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

    def match(self, img: np.ndarray):
        """
        matches face from image against embeddings of smpls and returns best match
        """
        face = self.detector.detect_face(img)
        if face is None or any(i == 0 for i in face.shape):
            print("could not detect face")
            return
        face_repr = self.model(face)
        scores = []
        skipped = []
        for fpath, smpl_repr in self.smpls_embeddings.items():
            if smpl_repr is None:
                skipped.append(fpath)
                continue
            assert face_repr.shape == smpl_repr.shape
            scores.append(Distance(smpl_repr, face_repr).euclidean_l2())
        fnames = list(self.smpls_embeddings.keys())
        smpl = cv2.imread(fnames[np.argmin(scores)])
        print(face.shape)
        merged = merge(face, smpl)
        fname = sha256(merged).hexdigest()
        cv2.imwrite(f"results/ibug/{fname}.png", merged)
        if not self.headless:
            show_comparison_cv(img, smpl, final=True)
