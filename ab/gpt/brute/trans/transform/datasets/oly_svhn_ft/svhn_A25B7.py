import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomEqualize(p=0.48), transforms.RandomPosterize(bits=6, p=0.72), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.4099, 0.4098, 0.4071], [0.1745, 0.1747, 0.1744])])