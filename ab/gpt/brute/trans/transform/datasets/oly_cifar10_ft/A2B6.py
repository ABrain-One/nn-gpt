<import torch</import>
<import torchvision.transforms as transforms</import>
<define>
def transform(norm):
    return transforms.Compose([
        transforms.RandomPosterize(bits=6, p=0.38),
        transforms.RandomGrayscale(p=0.71),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
     ])
</define>