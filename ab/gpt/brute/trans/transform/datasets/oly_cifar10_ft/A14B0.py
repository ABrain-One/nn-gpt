import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(9, 91, 57), padding_mode='reflect'),
    transforms.RandomVerticalFlip(p=0.47),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.39, 1.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])