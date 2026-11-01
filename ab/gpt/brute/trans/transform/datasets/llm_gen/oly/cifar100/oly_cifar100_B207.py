import torch<br/>
import torchvision.transforms as transforms<br/>
&nbsp;<br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(32, scale=(0.8, 1.2), ratio=(1.0, 1.0)),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(),<br/>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(hue=0.1, saturation=0.1),<br/>
&nbsp; &nbsp; &nbsp; transforms.Pad(padding=4, fill=(120, 111, 102), padding_mode='symmetric'),<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize(64),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(*norm)<br/>
&nbsp; &nbsp; ])