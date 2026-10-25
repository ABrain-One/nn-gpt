import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(15, 198, 213), padding_mode='reflect'), transforms.RandomSolarize(threshold=225, p=0.27), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.4969, 0.4848, 0.4909], std=[0.202, 0.2013, 0.2011])])