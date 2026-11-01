import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.32, 1.11)),
    transforms.RandomSolarize(threshold=16, p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
