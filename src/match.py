import cv2
import numpy as np
import time

from typing import Any
from deepface import DeepFace
from src.utils import write_file, merge
from src.data import get_smpls
from src.visualization import show_comparison_cv


class Matcher:

    def __init__(
        self,
        headless: bool,
        face: np.ndarray,
        face_name: str,
        model: Any,
        model_name: Any,
        detector_backend: str,
    ):
        self.headless = headless
        self.face = face
        self.face_name = face_name
        self.model = model
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.scores = []
        self.res = []
        self.inference_times = []
        self.paths, self.smpls = get_smpls("data/smpls")
        self.smpl_zip = enumerate(zip(self.paths, self.smpls))

    def log_stats(self) -> int:
        best_score_idx = np.argmin(self.scores)
        best_match = self.scores[best_score_idx]
        landed = np.count_nonzero(np.array(self.scores) != 1)
        print("\ndetection rate: %d/%d" % (landed, len(self.scores)))
        print("best match: %.2f" % best_match)
        if len(self.inference_times):
            print("average time per image: %.2fs" %
                  np.mean(self.inference_times))
        return best_score_idx

    def summarize(self):
        print("total time elapsed: %.2fs" % float(self.toc - self.tic))
        best_score_idx = self.log_stats()
        smpl = self.smpls[best_score_idx]
        merged = merge(self.face, smpl)
        base_path = f"results/{self.detector_backend}/{self.model_name}"
        fpath = f"{base_path}/json/{self.face_name.replace('.jpg', '')}.json"
        write_file(self.res, path=fpath)
        cv2.imwrite(f"{base_path}/image/{self.face_name}.png", merged)
        print("saved img %s" % self.face_name)
        if not self.headless:
            show_comparison_cv(self.face, smpl, final=True)

    def match(self):
        self.tic = time.time()
        # TODO add tqdm here
        for idx, [path, smpl] in self.smpls_zip:
            try:
                inference_tic = time.time()
                result = DeepFace.verify(
                    img1_path=self.face,
                    img2_path=smpl,
                    model=self.model,
                    detector_backend=self.detector_backend,
                )
                inference_toc = time.time()
                inference_time = float(inference_toc - inference_tic)
                self.inference_times.append(inference_time)
                self.res.append({path: result})
                if not self.headless:
                    show_comparison_cv(self.face, smpl)
                distance = result['distance']
                self.scores.append(distance)
                if idx % 10 == 1:
                    self.log_stats()
            except:
                self.scores.append(1)
        self.toc = time.time()
        self.summarize()
