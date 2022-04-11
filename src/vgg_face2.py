import onnxruntime as ort
import numpy as np
import cv2

from typing import List


class VGGFace2:

    def __init__(self):
        self.session = ort.InferenceSession("models/vggface2.onnx")
        self.input_shape = self.session.get_inputs()[0].shape
        print("vggface2 session started")

    def __call__(self, img: np.ndarray):
        inp = self.preprocess(img)
        out = self.feed(inp)
        return self.postprocess(out)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        :param img: the face to be fed into the vgg model

        this is assuming that the face has been detected,
        preprocessing is model specific here and is mostly
        around getting the input size right
        """
        _img = img.copy()
        _img = self.apply_padding(_img)
        _img = np.transpose(_img, (2, 1, 0))
        _img = np.expand_dims(_img, axis=0)
        _img = _img / 255
        _img = _img.astype(np.float32)
        return _img

    def feed(self, inp: np.ndarray):
        out = self.session.run(["output_1"], {"input_1": inp})
        return out

    def postprocess(self, out: List):
        [[embedding]] = out
        return embedding

    def apply_padding(self, img: np.ndarray) -> np.ndarray:
        _img = img.copy()
        _, _, h, w = self.input_shape
        factor_0 = h / _img.shape[0]
        factor_1 = w / _img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(_img.shape[1] * factor), int(_img.shape[0] * factor))
        _img = cv2.resize(_img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = h - _img.shape[0]
        diff_1 = w - _img.shape[1]
        _img = np.pad(
            _img,
            (
                (diff_0 // 2, diff_0 - diff_0 // 2),
                (diff_1 // 2, diff_1 - diff_1 // 2),
                (0, 0),
            ),
            'constant',
        )
        return _img
