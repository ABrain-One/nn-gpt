import torchvision.transforms as T

def transform(norm):
    return T.Compose([
    T.RandAugment(num_ops=2, magnitude=9),
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(*norm)
])