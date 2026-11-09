import torch<br/>
import torchvision.transforms as transforms<br/>
<br/>
def transform(norm):<br/>
    return transforms.Compose([<br/>
        transforms.RandomHorizontalFlip(p=0.56),<br/>
        transforms.Pad(padding=4, fill=(132, 23, 78), padding_mode='constant'),<br/>
        transforms.RandomCrop(29),<br/>
        transforms.RandomRotation(10),<br/>
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
        transforms.Resize((64,64)),<br/>
        transforms.ToTensor(),<br/>
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267,0.256,0.276])<br/>
    ])