import tensorflow as tf


class TestGPU(tf.test.TestCase):

    def test_works_with_gpu(self):
        self.assertTrue(tf.test.is_gpu_available())
