import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.Pad(padding=1, fill=(22, 220, 159), padding_mode='edge'), transforms.GaussianBlur(kernel_size=5, sigma=(0.6, 1.7)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.27, 0.29, 0.27], std=[0.3, 0.3, 0.31])])