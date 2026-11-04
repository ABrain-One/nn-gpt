import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(112, 21, 113), padding_mode='edge'), transforms.GaussianBlur(kernel_size=5, sigma=(0.39, 1.82)), transforms.CenterCrop(size=29), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.29, 0.282, 0.2939], std=[0.278, 0.285, 0.2606])])