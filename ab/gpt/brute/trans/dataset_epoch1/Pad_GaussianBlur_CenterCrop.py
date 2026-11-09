import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(112, 21, 113), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.39, 1.81)),
    transforms.CenterCrop(size=25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
