import torch

from facenet_pytorch import InceptionResnetV1


def export_to_onnx(model: torch.Module):
    # most have the same input size (224, 224)
    model = model.cuda().eval()
    dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
    torch.onnx.export(model, dummy_input, "models/vggface2.onnx", verbose=True)


if __name__ == "__main__":
    model = InceptionResnetV1(pretrained='vggface2')
    export_to_onnx(model)
