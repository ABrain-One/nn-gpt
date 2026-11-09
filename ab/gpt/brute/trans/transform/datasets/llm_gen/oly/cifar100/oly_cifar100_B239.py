import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
    return transforms.Compose([<br/>
        transforms.Resize(72),<br/>
        transforms.RandomCrop(64),<br/>
        transforms.RandomApply([transforms.VerticalFlip()], p=0.2),<br/>
        transforms.RandomApply([transforms.HorizontalFlip()], p=0.5),<br/>
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),<br/>
        transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=0.75),<br/>
        transforms.ToTensor(),<br/>
        transforms.Normalize(*norm)<br/>
    ])