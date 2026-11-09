import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.65),
    transforms.RandomEqualize(p=0.11),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.96, 1.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
