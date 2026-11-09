import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.65),
    transforms.RandomCrop(size=29),
    transforms.RandomPerspective(distortion_scale=0.16, p=0.27),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
