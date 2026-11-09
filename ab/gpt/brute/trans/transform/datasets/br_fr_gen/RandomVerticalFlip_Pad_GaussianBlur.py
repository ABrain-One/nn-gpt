import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.79),
    transforms.Pad(padding=1, fill=(30, 142, 106), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.71, 1.52)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
