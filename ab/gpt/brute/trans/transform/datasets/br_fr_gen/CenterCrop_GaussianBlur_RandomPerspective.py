import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.19, 1.82)),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
