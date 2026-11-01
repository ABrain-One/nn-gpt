import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(211, 164, 118), padding_mode='constant'), transforms.RandomInvert(p=0.1), transforms.CenterCrop(size=30), transforms.RandomRotation(degrees=15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.279])])