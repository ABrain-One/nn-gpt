import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.RandomGrayscale(p=0.33), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.276, 0.2821, 0.2774], [0.2881, 0.2902, 0.2821])])