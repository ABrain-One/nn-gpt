<name>TransformModule</name>
    <src>import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(219, 105, 67), padding_mode='edge'),
    transforms.RandomPosterize(bits=4, p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.47, 0.51, 0.46], std=[0.93, 0.92, 0.85], dtype=torch.float32),
])
</src>