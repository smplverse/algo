import os
import torch
import onnx

from facenet_pytorch import InceptionResnetV1


def export_to_onnx(model: torch.nn.Module, model_name: str):
    # most have the same input size (224, 224)
    torch_model = model.cuda().eval()
    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
    onnx_model_name = f"{model_name}.onnx"
    if onnx_model_name in os.listdir('models'):
        print("onnx model already exists")
        return
    torch.onnx.export(
        torch_model,
        dummy_input,
        f"models/{onnx_model_name}",
        input_names=["input_1"],
        output_names=["output_1"],
    )
    onnx_model = onnx.load(f"models/{onnx_model_name}")
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('err: %s' % e)
    else:
        print('ok')


if __name__ == "__main__":
    model_name = 'vggface2'
    model = InceptionResnetV1(pretrained=model_name)
    export_to_onnx(model, model_name)
