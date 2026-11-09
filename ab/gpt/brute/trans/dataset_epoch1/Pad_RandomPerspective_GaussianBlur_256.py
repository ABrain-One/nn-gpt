import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(245, 165, 98), padding_mode='edge'),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.77),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.89, 1.9)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
