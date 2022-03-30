import pytest


def test_torch_works_with_cuda():
    import torch
    assert torch.cuda.is_available() == True


def test_tensorrt_imports():
    import tensorrt as trt
    assert tensorrt is not None


def test_pycuda_works():
    import pycuda.gpuarray as gpuarray
    import numpy.linalg as la
    a = np.arange(200000, dtype=np.float32)
    b = a + 17

    a_g = gpuarray.to_gpu(a)
    b_g = gpuarray.to_gpu(b)
    diff = (a_g - 3 * b_g + (-a_g)).get() - (a - 3 * b + (-a))
    assert la.norm(diff) == 0

    diff = (a_g * b_g).get() - a * b
    assert la.norm(diff) == 0
