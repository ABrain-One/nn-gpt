import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomSolarize(threshold=241, p=0.17), transforms.RandomApply([transforms.CenterCrop(24), transforms.GaussianBlur(kernel_size=3, sigma=(0.33, 1.05)), transforms.Resize(64)], p=0.5), transforms.Pad(padding=3, fill=(42, 138, 150)), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])