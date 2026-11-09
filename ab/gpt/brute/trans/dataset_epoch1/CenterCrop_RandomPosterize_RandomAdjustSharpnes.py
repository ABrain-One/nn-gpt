import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.RandomPosterize(bits=6, p=0.64),
    transforms.RandomAdjustSharpness(sharpness_factor=1.98, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
