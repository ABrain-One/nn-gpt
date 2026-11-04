import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.86), transforms.RandomGrayscale(p=0.68), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.294, 0.2829, 0.2903], std=[0.2611, 0.2554, 0.2512])])