import cv2
import numpy as np
import time

from hashlib import sha256
from mediapipe.python.solutions.face_detection import FaceDetection
from distance import Distance
from src.utils import merge
from src.visualization import show_comparison_cv
from typing import Any, Union, List


class Matcher:

    def __init__(
        self,
        headless: bool,
        model: Any,
        session: bool = False,
    ):
        self.headless = headless
        self.session = session
        self.model = model
        self.face_detector = FaceDetection(min_detection_confidence=0.7)
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

    @staticmethod
    def align(
        img: np.ndarray,
        left_eye: List[int],
        right_eye: List[int],
    ):
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        if left_eye_y > right_eye_y:
            point_3rd = np.array([right_eye_x, left_eye_y])
            direction = -1
        else:
            point_3rd = np.array([left_eye_x, right_eye_y])
            direction = 1

        a = Distance(left_eye, point_3rd).euclidean()
        b = Distance(right_eye, point_3rd).euclidean()
        c = Distance(right_eye, left_eye).euclidean()
        if b != 0 and c != 0:
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)
            angle = (angle * 180) / np.pi
            if direction == -1:
                angle = 90 - angle
            img = img.rotate(direction * angle)
        return img

    def detect_face(self, img: np.ndarray) -> Union[np.ndarray, None]:
        results = self.face_detector.process(img)
        if not len(results.detections):
            return None
        [det, *_] = results.detections
        bbox = det.location_data.relative_bounding_box
        landmarks = det.location_data.relative_keypoints
        _, img_height, img_width = img.shape
        x = int(bbox.xmin * img_width)
        w = int(bbox.width * img_width)
        y = int(bbox.ymin * img_height)
        h = int(bbox.height * img_height)
        right_eye = np.array([
            landmarks[0].x * img_width,
            landmarks[0].y * img_height,
        ]).astype(int)
        left_eye = np.array([
            landmarks[1].x * img_width,
            landmarks[1].y * img_height,
        ]).astype(int)
        face = img[y:y + h, x:x + w]
        face = self.align(face, left_eye, right_eye)
        return face

    def match(self, img: np.ndarray, smpl: np.ndarray):
        inference_tic = time.time()

        face = self.detect_face(img)
        smpl_face = self.detect_face(smpl)

        face_repr = self.model(face)
        smpl_repr = self.model(smpl_face)
        assert face_repr.shape == smpl_repr.shape

        distance = Distance(smpl_repr, face_repr).cosine()

        inference_toc = time.time()
        if not self.headless:
            show_comparison_cv(face, smpl)
        if self.session:
            inference_time = float(inference_toc - inference_tic)
            self.inference_times.append(inference_time)
            self.scores.append(distance)
