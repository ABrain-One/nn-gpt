import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(24, 95, 143), padding_mode='symmetric'), transforms.GaussianBlur(kernel_size=5, sigma=(0.79, 1.41)), transforms.RandomSolarize(threshold=194, p=0.14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])