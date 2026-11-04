import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(76, 246, 30), padding_mode='symmetric'), transforms.RandomAutocontrast(p=0.71), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.2661, 0.298, 0.257], std=[0.2874, 0.2939, 0.27])])