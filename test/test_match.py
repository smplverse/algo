from deepface import DeepFace


def test_performs_single_match():
    model = DeepFace.build_model('VGG-Face')
    res = DeepFace.verify(
        img1_path="data/input/AJ_Cook_0001.jpg",
        img2_path="data/input/AJ_Cook_0001.jpg",
        model=model,
    )
    assert round(res['distance'], 5) == 0
