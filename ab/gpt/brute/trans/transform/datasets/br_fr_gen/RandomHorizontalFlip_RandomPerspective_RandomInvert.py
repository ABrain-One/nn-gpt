import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.83),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.19),
    transforms.RandomInvert(p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
