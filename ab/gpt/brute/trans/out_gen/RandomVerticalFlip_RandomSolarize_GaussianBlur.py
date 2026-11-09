import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.28),
    transforms.RandomSolarize(threshold=13, p=0.73),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.65, 1.89)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
