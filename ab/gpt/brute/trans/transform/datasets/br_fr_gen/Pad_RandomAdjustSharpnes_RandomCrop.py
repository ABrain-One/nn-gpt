import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(26, 133, 164), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.89, p=0.25),
    transforms.RandomCrop(size=27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
