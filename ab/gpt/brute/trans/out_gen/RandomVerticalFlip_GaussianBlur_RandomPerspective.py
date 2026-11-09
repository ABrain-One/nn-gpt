import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.73),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.48, 1.79)),
    transforms.RandomPerspective(distortion_scale=0.29, p=0.79),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
