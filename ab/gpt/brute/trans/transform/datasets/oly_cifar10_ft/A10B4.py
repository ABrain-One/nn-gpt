<name>transform</name>
    <code>import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.85),
    transforms.Pad(padding=0, fill=(124, 58, 49), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.37, p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
</code>