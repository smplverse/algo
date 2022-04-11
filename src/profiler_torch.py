import torch

from facenet_pytorch import InceptionResnetV1
from torch.profiler import profile, ProfilerActivity

model = InceptionResnetV1(pretrained="vggface2")
model = model.eval().cuda()

inputs = torch.randn(1, 3, 224, 224, device="cuda")

with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        profile_memory=True,
) as prof:
    model(inputs)

print(prof.key_averages().table())
