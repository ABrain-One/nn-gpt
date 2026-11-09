import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(9, 91, 212), padding_mode='reflect'),
    transforms.RandomVerticalFlip(p=0.47),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.87, 1.15)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
