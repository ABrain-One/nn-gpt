import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.58),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.12, 1.11)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
