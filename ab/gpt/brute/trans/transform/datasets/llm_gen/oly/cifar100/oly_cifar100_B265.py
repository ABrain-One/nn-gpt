import torch<br/>
from torchvision import transforms<br/>
&nbsp;<br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomApply([<br/>
&nbsp; &nbsp; &nbsp; &nbsp; transforms.GaussianBlur(3),<br/>
&nbsp; &nbsp; &nbsp; ], p=0.25),<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize((224, 224)),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])<br/>
&nbsp; &nbsp; ])