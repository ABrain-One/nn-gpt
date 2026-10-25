import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.78),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.31, 1.55)),
    transforms.RandomSolarize(threshold=151, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
