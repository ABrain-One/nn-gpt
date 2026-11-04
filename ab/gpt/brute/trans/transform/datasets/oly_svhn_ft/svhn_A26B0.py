import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.Pad(padding=4, fill=(173, 214, 195), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.2801, 0.2772, 0.2748], [0.0276, 0.0273, 0.0275]), transforms.Resize((64, 64)), transforms.Normalize([0.2801, 0.2772, 0.2748], [0.0276, 0.0273, 0.0275])])