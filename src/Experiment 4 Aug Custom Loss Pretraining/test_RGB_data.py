import torch
import cv2
from Aug import Aug_Hyperspectral_Data
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

image_height = 512
image_width = 512

contrast_values = [i*0.1 for i in range(-1, 3) if i!=0]
brightness_values = [i*0.1 for i in range(-1, 2) if i!=0]
rgb_shift_values = [i*10 for i in range(-1, 1) if i!=0]

def get_augmentation_composition(image_height, image_width,specific_augmentation= None):
    """Get Augmentation Composition

    Args:
        specific_augmentation (Albumentation, optional): specific augmentation to be applied to compositions. Defaults to None.

    Returns:
        aug: Composition of augmentation
    """
    list_of_augmentation = []
    last_augmentation = [A.Resize(image_height, image_width), A.Normalize(), ToTensorV2()]
    if specific_augmentation:
        list_of_augmentation.append(specific_augmentation)
    list_of_augmentation.extend(last_augmentation)
    aug = A.Compose(list_of_augmentation)
    return aug

def get_all_compositions(image_height, image_width):
    """Get list of compositions

    Returns:
        list_of_compositions: list of compositions
    """
    list_of_compositions = []
    for elem in contrast_values:
        list_of_compositions.append(get_augmentation_composition(image_height, image_width,A.augmentations.transforms.RandomContrast(limit=[elem, elem], always_apply= True, p=1)))

    for elem in brightness_values:
        list_of_compositions.append(get_augmentation_composition(image_height, image_width, A.augmentations.transforms.RandomBrightness(limit=[elem, elem], always_apply= True, p=1)))
    
    for elem in rgb_shift_values:
        list_of_compositions.append(get_augmentation_composition(image_height, image_width, A.augmentations.transforms.RGBShift(r_shift_limit=[elem, elem] , g_shift_limit=0, b_shift_limit=0, always_apply=True, p=1)))

    for elem in rgb_shift_values:
        list_of_compositions.append(get_augmentation_composition(image_height, image_width, A.augmentations.transforms.RGBShift(r_shift_limit=0 , g_shift_limit=[elem, elem], b_shift_limit=0, always_apply=True, p=1)))

    for elem in rgb_shift_values:
        list_of_compositions.append(get_augmentation_composition(image_height, image_width, A.augmentations.transforms.RGBShift(r_shift_limit=0 , g_shift_limit=0, b_shift_limit=[elem, elem], always_apply=True, p=1)))    
    
    list_of_compositions.append(get_augmentation_composition(image_height, image_width))
    return list_of_compositions


class RGBDataset(Dataset):
    def __init__(self, dataframe, shape = (image_height, image_width), transform=None):
        self.image_height = image_height
        self.image_width = image_width
        self.dataframe = dataframe
        self.shape = shape
        self.transform = transform
        self.augmentation_list = get_all_compositions(self.image_height, self.image_width)
        if self.transform ==None:
            self.Aug = Aug_Hyperspectral_Data(self.shape, list_augmentations = [])
        else:
            self.Aug = Aug_Hyperspectral_Data(self.shape, list_augmentations = self.transform )
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        data_path = self.dataframe["path"].iloc[idx]  # Assuming "path" column is the first column
        label = self.dataframe["label"].iloc[idx]  # Assuming "label" column is the second column
        img = cv2.imread(data_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_list = []
        
        for i, transform in enumerate(self.augmentation_list):
            transformed = transform(image=img.copy())
            transformed_image = transformed["image"]
            image_list.append(transformed_image)

        data = torch.vstack(image_list)
        if self.transform is not None:
            data = self.Aug(data)

        return data, label
    
    
    
df = pd.read_csv("rgb_data.csv")

dataset = RGBDataset(df)

for data, label in dataset:
    print(data.shape, label)
    break