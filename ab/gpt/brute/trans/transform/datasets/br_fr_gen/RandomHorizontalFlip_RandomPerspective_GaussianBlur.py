import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.75),
    transforms.RandomPerspective(distortion_scale=0.14, p=0.17),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.88, 1.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
