from deepface import DeepFace


def test_performs_single_match():
    res = DeepFace.verify(img1_path="data/input/AJ_Cook_0001.jpg",
                          img2_path="data/input/AJ_Cook_0001.jpg")
    assert round(res['distance'], 5) == 0
