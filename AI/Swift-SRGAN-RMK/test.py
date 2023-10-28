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
import torchvision
import tqdm
from torchvision.transforms.functional import to_tensor
import cv2
import time

from PIL import *

imsize = 256
loader = transforms.ToTensor()

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

#def image_loader(image_name):
#    """load image, returns cuda tensor"""
#    image = Image.open(image_name)
#    image = loader(image).float()
#    #image = Variable(image, requires_grad=True)
#    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#    return image.cuda()  #assumes that you're using GPU

display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor()
])

def main(opt):
    mean = [0.4207, 0.4651, 0.3286]
    std = [0.2025, 0.1796, 0.1711]
    
    model = torch.jit.load('./test/optimized_model.pt')
    model.eval() 
    
    image_transforms = transforms.transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
    
    image = Image.open('./test/LR/lr.png')
    
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    image = image.cuda()
    
    # Start timer
    start_time = time.time()
    resultado = model(image)
    # End timer
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time) 
    
    #resultado = display_transform(resultado.squeeze(0))
    torchvision.utils.save_image(
                    resultado,
                    "./test/Upscaled/upscaledResult.png",
                    padding=5,
               )
    
    #resultado = resultado.squeeze(0)
    #image = torchvision.utils.make_grid(image, nrow=3, padding=5)
    #resultado.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swift-SRGAN')
    opt = parser.parse_args()
    main(opt)
    


"""
    #convert = transforms.ToPILImage()
    
    display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
    
    image = Image.open('./test/LR/lr.png').convert('RGB')
    #image = to_tensor(image)
    image = image.cuda()
    #image = image.unsqueeze(0)
    image = display_transform(image)
    #image = tqdm.tqdm(image)
    image = image.unsqueeze(0)
    
    #image = image_loader('./test/LR/lr.png')
    image = model(image)
    
    torchvision.utils.save_image(
                    image,
                    "./test/Upscaled/upscaledResult.png",
                    padding=5,
                )
    
    #image.to('cuda')
    #image = image.size(0)
    #image = image.squeeze(0)
    #image = convert(image)
    #image.show()
"""