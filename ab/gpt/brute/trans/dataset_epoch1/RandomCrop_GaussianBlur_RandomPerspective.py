import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.37, 1.51)),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
