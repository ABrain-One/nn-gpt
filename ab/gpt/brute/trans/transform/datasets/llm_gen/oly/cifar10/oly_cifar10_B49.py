import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp; &nbsp; return transforms.Compose([<br>
&nbsp; &nbsp; &nbsp; transforms.RandomPerspective(distortion_scale=0.2, p=0.2),<br>
&nbsp; &nbsp; &nbsp; transforms.Pad(padding=5, fill=(255, 69, 120), padding_mode='constant'),<br>
&nbsp; &nbsp; &nbsp; transforms.Resize((64,64)),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomCrop(32),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomAffine(<br>
&nbsp; &nbsp; &nbsp; &nbsp; degrees=15,<br>
&nbsp; &nbsp; &nbsp; &nbsp; translate=(0.1, 0.2),<br>
&nbsp; &nbsp; &nbsp; &nbsp; scale=(0.8, 1.2),<br>
&nbsp; &nbsp; &nbsp; &nbsp; shear=(-5, 5)<br>
&nbsp; &nbsp; &nbsp; ),<br>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br>
&nbsp; &nbsp; &nbsp; transforms.Normalize(**norm)<br>
&nbsp; &nbsp; ])