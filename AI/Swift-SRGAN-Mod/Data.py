import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import Dataset
from torchvision.utils import save_image

def is_file_image(file_name):
    return file_name.endswith("png" or "jpg" or "jpeg" or "PNG" or "JPG" or "JPEG")

def transformImageHR(crop_size):
    transformation = transforms.Compose([transforms.RandomCrop(crop_size, pad_if_needed = True), transforms.RandomHorizontalFlip(0.33), transforms.RandomVerticalFlip(0.33), transforms.ToTensor()])
    return transformation

def transformImageLR(crop_size, upscale_factor):
    transformation = transforms.Compose([transforms.ToPILImage(), transforms.Resize(crop_size // upscale_factor, functional.InterpolationMode("bicubic")), transforms.ToTensor()])
    return transformation
    
# takes a image dataset and creates HR and LR training versions of its images and stores them as Tensors inside the Dataset Type
class GenerateTrainingDataset(Dataset):
    def __init__(self, dataset_dir, crop_size = 1024, upscale_factor = 4):
        super().__init__()
        self.crop_size = (crop_size - crop_size % upscale_factor)
        
        self.hr_tranformation = transformImageHR(self.crop_size)
        self.lr_tranformation = transformImageLR(self.crop_size, upscale_factor)
        
        self.original_image_filenames = [os.path.join(dataset_dir, image_name) for image_name in os.listdir(dataset_dir) if is_file_image(image_name)]
        
    def __getitem__(self, index):
        original_image = Image.open(self.original_image_filenames[index]).convert('RGB')
        hr_image = self.hr_tranformation(original_image)
        lr_image = self.lr_tranformation(hr_image)
        return (lr_image, hr_image)
    
    def __len__(self):
        lenght = len(self.original_image_filenames)
        return lenght