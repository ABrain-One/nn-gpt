import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(213, 181, 213), padding_mode='symmetric'), transforms.RandomSolarize(threshold=221, p=0.37), transforms.GaussianBlur(kernel_size=5, sigma=(1.19, 1.47)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])