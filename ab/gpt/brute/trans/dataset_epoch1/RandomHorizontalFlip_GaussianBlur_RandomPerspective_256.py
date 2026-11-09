import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.14),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.38, 1.41)),
    transforms.RandomPerspective(distortion_scale=0.24, p=0.67),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
