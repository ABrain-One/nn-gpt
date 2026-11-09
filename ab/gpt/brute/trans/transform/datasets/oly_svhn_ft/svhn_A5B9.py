import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(167, 120, 131), padding_mode='reflect'), transforms.RandomSolarize(threshold=152, p=0.15), transforms.GaussianBlur(kernel_size=5, sigma=(0.32, 1.51)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])