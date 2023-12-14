import torch
import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt
import random

class Aug_Hyperspectral_Data():
    """ """
    def __init__(self, size_tuple, list_augmentations= ["resize","definition_loss", "rotation", 
                                                        "translation", "cut_out", 
                                                        "horizontal_flip", "vertical_flip"]):
        super().__init__()
    
        self.size_tuple= size_tuple
        self.scale= random.uniform(0.3, 1)
        self.trans_x = int((self.size_tuple[0]) / random.choice([i for i in range(7, 15)])) * random.choice([-1,1])
        self.trans_y = int((self.size_tuple[1]) / random.choice([i for i in range(7, 15)])) * random.choice([-1,1])
        self.zoom_val = random.uniform(0.8, 1.3)
        self.pad_min = 10
        self.pad_max = 100
        self.probability = 0.5
        self.angle = random.randint(0, 360)
        
        self.method_mapping = self.get_mapping()
        self.list_augmentations = list_augmentations
        self.augmentation_list = [self.method_mapping[aug] for aug in self.list_augmentations]
        self.set_cutout_params()
    def get_mapping(self):
        """ """
        return {
            "resize": self.resize,
            "definition_loss": self.definition_loss,
            
            "rotation": self.rotation,
            "translation": self.translation,
            "cut_out": self.cut_out,
            
            "shuffle_channels": self.shuffle_channels,
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
        }
        
    def set_cutout_params(self):
        """ """
        self.start_point_x = random.randint(self.pad_max, 512-self.pad_max)
        self.start_point_y = random.randint(self.pad_max, 512-self.pad_max)
        self._height = random.randint(self.pad_min, self.pad_max)
        self._width = random.randint(self.pad_min, self.pad_max)
    
    def definition_loss(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """
        scaled_image_dim_0, scaled_image_dim_1 = int(self.size_tuple[0] * self.scale), int(self.size_tuple[1] * self.scale) 
        array = cv2.resize(array, dsize=(scaled_image_dim_0, scaled_image_dim_1), interpolation=cv2.INTER_CUBIC)
        array = cv2.resize(array, dsize=(self.size_tuple[0], self.size_tuple[1]), interpolation=cv2.INTER_CUBIC) 
        return array

    def rotation(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """
        image_center = tuple(np.array(array.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.0)
        array = cv2.warpAffine(array, rot_mat, dsize=(self.size_tuple[0], self.size_tuple[1]), flags=cv2.INTER_LINEAR)
        return array
    
    def resize(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """

        array = cv2.resize(array, (self.size_tuple[0], self.size_tuple[1]))
        return array
 
    def translation(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """
        translation_matrix = np.array([[1, 0, self.trans_x],[0, 1, self.trans_y]], dtype=np.float32)
        array = cv2.warpAffine(src=array, M=translation_matrix, dsize=(self.size_tuple[0], self.size_tuple[1]))
        return array
    
    def cut_out(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """
        color = (0, 0, 0)
        thickness = -1
        start_point = (self.start_point_x, self.start_point_y)
        end_point = (self.start_point_x + self._width, self.start_point_y + self._height)
        
        img_list = [array[:,:, range(i*3,(i+1)*3)] for i in range(9)]
        img_stack = [cv2.rectangle(img.copy(), start_point, end_point, color, thickness) for img in img_list]

        array = np.concatenate(img_stack, axis = -1)
        return array  
    
    def shuffle_channels(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """
        num_channels = array.shape[2]

        # Generate a random permutation of channel indices
        channel_indices = np.random.permutation(num_channels)

        # Shuffle the channels based on the permutation
        shuffled_image_array = array[:, :, channel_indices]
        return shuffled_image_array
    
    def horizontal_flip(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """
        random_number = np.random.random()
        if random_number < self.probability:
            array = np.flip(array, axis =0)
            return np.flip(array, axis =0)
        else: 
            return array
    
    def vertical_flip(self, array):
        """

        Parameters
        ----------
        array :
            

        Returns
        -------

        """
        
        random_number = np.random.random()
        if random_number < self.probability:
            array = np.flip(array, axis =1)
            
            return array
        else:
            return array
    
    def __call__(self, array):
        for aug in self.augmentation_list:
            array =aug(array)
        return array
