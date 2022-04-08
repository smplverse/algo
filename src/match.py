import cv2
import numpy as np
import time

from tqdm import tqdm

from typing import Any
from distance import Distance
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
    ):
        self.headless = headless
        self.face = face
        self.face_name = face_name
        self.model = model
        self.distance = Distance()
        self.scores = []
        self.res = []
        self.inference_times = []
        self.paths, self.smpls = get_smpls("data/smpls")
        self.smpls_zip = zip(self.paths, self.smpls)

    def summarize(self):
        print("total time elapsed: %.2fs" % float(self.toc - self.tic))
        best_score_idx = np.argmin(self.scores)
        best_match = self.scores[best_score_idx]
        landed = np.count_nonzero(np.array(self.scores) != 1)
        print("\ndetection rate: %d/%d" % (landed, len(self.scores)))
        print("best match: %.2f" % best_match)
        if len(self.inference_times):
            print("average time per image: %.2fs" %
                  np.mean(self.inference_times))
        smpl = self.smpls[best_score_idx]
        merged = merge(self.face, smpl)
        just_name = self.face_name.replace('.jpg', '')
        fpath = f"results/json/{just_name}.json"
        write_file(self.res, path=fpath)
        cv2.imwrite(f"results/image/{just_name}.png", merged)
        print("saved img %s" % self.face_name)
        if not self.headless:
            show_comparison_cv(self.face, smpl, final=True)

    def match(self):
        self.tic = time.time()
        for _ in tqdm(range(len(self.smpls))):
            path, smpl = self.smpls_zip.__next__()
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
            except:
                self.scores.append(1)
        self.toc = time.time()
        self.summarize()
