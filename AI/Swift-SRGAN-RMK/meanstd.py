import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
              transforms.ToTensor()
            ])

image = Image.open('./test/LR/lr.png')
image = transform(image)
mean = torch.mean(image, dim=(1, 2))
std = torch.std(image, dim=(1, 2))
print(mean)
print(std)