import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(244, 7, 205), padding_mode='constant'), transforms.CenterCrop(29), transforms.RandomPosterize(5, p=0.25), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.4865, 0.4308], std=[0.2673, 0.2565, 0.2762])])