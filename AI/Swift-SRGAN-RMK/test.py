import torch
from models import Generator
import argparse
import PIL
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from PIL import *

imsize = 256
loader = transforms.ToTensor()

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

def main(opt):
    model = torch.jit.load('./test/optimized_model.pt')
    model.eval()
    
    image = image_loader('./test/LR/lr.png')
    image = model(image) 
    image.to('cuda')
    convert = transforms.ToPILImage()
    image = image.squeeze(0)
    image = convert(image)
    image.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swift-SRGAN')
    opt = parser.parse_args()
    main(opt)