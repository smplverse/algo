import cv2
import torch


def test_match_face():
    from src.match import match_face
    img_path = "data/input/AJ_Cook/AJ_Cook_0001.jpg"
    img = cv2.imread(img_path)
    assert isinstance(match_face(img), torch.float)
