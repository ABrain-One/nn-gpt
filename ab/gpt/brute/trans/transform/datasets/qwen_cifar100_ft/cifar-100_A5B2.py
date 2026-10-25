import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(152, 112, 137), padding_mode='symmetric'), transforms.RandomGrayscale(p=0.68), transforms.GaussianBlur(kernel_size=5, sigma=(0.65, 1.87)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])