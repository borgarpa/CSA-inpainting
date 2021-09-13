import random
import torch
import cv2
import numpy as np
from PIL import Image
from glob import glob


class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, sent1_root, transform, normalization, data_cast):
        super(Data_load, self).__init__()
        self.transform = transform
        self.normalization = normalization
        self.data_cast = data_cast


        self.paths = glob('{:s}/*'.format(img_root),
                              recursive=True)

        self.s1_paths = glob('{:s}/*'.format(sent1_root),
                              recursive=True)

        self.mask_paths = glob('{:s}/*.png'.format(mask_root))

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        ### TODO: Change Image.open by either rasterio.open or np.load. Datatype is changed to uint8 during the transform operation
        # gt_img = rasterio.open(self.paths[index])
        
        gt_img = np.load(self.paths[index])
        # gt_img = self.transform(gt_img) ### Remove RGB transformation
        gt_img = self.data_cast[0](gt_img)
        gt_img = torch.cat(tuple(self.transform(gt_img[n, :, :]) for n in range(gt_img.size(0))), 0) # im in format CxHxW
        gt_img = self.normalization[0](gt_img)

        # sent1 = rasterio.open(self.s1_paths[index])
        sent1 = np.load(self.s1_paths[index])
        sent1 = self.data_cast[1](sent1)
        sent1 = torch.cat(tuple(self.transform(sent1[n, :, :]) for n in range(sent1.size(0))), 0)
        sent1 = self.normalization[1](sent1)


        # mask = Image.open(self.mask_paths[index]) #[random.randint(0, self.N_mask - 1)])
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE) #[random.randint(0, self.N_mask - 1)])
        mask = self.transform(mask) ### Remove RGB transformation
        
        return gt_img, mask, sent1

    def __len__(self):
        return len(self.paths)
