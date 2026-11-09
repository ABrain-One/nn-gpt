import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomAdjustSharpness(sharpness_factor=1.86, p=0.65),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.46),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
