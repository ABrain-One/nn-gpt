import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.79),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.85, 1.19)),
    transforms.RandomSolarize(threshold=72, p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
