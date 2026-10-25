import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.12),<br/>
&nbsp; &nbsp; &nbsp; transforms.Pad(5, fill=(201, 131, 30), padding_mode='edge'),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomAffine(<br/>
&nbsp; &nbsp; &nbsp; &nbsp; degrees=5,<br/>
&nbsp; &nbsp; &nbsp; &nbsp; translate=(0.1, 0.1),<br/>
&nbsp; &nbsp; &nbsp; &nbsp; scale=(0.9, 1.1),<br/>
&nbsp; &nbsp; &nbsp; &nbsp; shear=(-2.0, 2.0)<br/>
&nbsp; &nbsp; &nbsp; ),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomCrop(27),<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize(64),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(**norm)<br/>
&nbsp; &nbsp; ])