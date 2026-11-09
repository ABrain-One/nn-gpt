import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.29),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.61),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.28, 1.34)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
