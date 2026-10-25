tag. So the final answer is:

<	tr>import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.65, 0.99), ratio=(0.95, 1.76)),
    transforms.RandomCrop(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAutocontrast(p=0.15),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=1.5)], p=0.25),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
    ])