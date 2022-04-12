import numpy as np
"""
# below to not relevant for now
import pytest

@pytest.mark.skip()
def test_tensorrt_works():
    import tensorrt as trt
    assert trt.Builder(trt.Logger())


@pytest.mark.skip()
def test_pycuda_works():
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import numpy.linalg as la
    import numpy as np
    a = np.arange(200000, dtype=np.float32)
    b = a + 17

    a_g = gpuarray.to_gpu(a)
    b_g = gpuarray.to_gpu(b)
    diff = (a_g - 3 * b_g + (-a_g)).get() - (a - 3 * b + (-a))
    assert la.norm(diff) == 0

    diff = (a_g * b_g).get() - a * b
    assert la.norm(diff) == 0


def test_tensorflow_works_with_gpu():
    import tensorflow as tf
    assert tf.test.is_gpu_available()
"""


def test_onnxruntime_works_with_cuda():
    import onnxruntime as ort
    providers = [
        'CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
    ]
    session = ort.InferenceSession(
        "models/vggface2.onnx",
        providers=providers,
    )
    inp = np.random.rand(1, 3, 224, 224).astype(np.float32)
    [out] = session.run(None, {"input_1": inp})
    assert out.shape == (1, 512)


def test_opencv_works():
    import cv2
    img = cv2.imread("data/smpls/000008.png")
    assert len(img.shape) == 3
