import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.Pad(padding=1, fill=(215, 153, 31), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.99, 1.66)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
