import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomResizedCrop(64, scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333)),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomRotation(degrees=(-10, 10)),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),<br/>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br/>
&nbsp;&nbsp;&nbsp;transforms.Normalize(**norm)<br/>
&nbsp;&nbsp;])