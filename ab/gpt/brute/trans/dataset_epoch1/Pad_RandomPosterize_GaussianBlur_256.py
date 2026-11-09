import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(46, 69, 110), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=4, p=0.11),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.42, 1.57)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
