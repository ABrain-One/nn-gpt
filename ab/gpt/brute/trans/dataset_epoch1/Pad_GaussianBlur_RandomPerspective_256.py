import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(19, 214, 145), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.27, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.74),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
