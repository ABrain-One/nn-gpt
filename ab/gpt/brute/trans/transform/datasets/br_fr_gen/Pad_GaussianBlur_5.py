import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(198, 245, 56), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.7, 1.33)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
