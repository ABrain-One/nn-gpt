import torch<br>
from torchvision import transforms<br>
&nbsp;<br>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(),<br>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])<br>
&nbsp;&nbsp;])<br>