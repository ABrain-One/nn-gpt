import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(31), transforms.ColorJitter(brightness=0.88, contrast=1.03, saturation=1.12, hue=0.09), transforms.RandomRotation(20), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762])])