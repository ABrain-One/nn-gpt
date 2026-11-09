import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.55),
    transforms.RandomSolarize(threshold=192, p=0.77),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.65, 1.4)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
