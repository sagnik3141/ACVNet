import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

class KITTIDatasetSPCSingle(Dataset):
    def __init__(self, datapath, list_filename, num_frames, training):
        self.datapath = datapath
        #self.datapath_12 = kitti12_datapath
        self.num_frames = num_frames
        #self.num_frames_med = num_frames_med
        #self.num_frames_lg = num_frames_lg
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        img = np.load(filename)
        img = np.mean(img[128-self.num_frames//2:129+self.num_frames], axis=0)
        #img_med = np.mean(img[128-self.num_frames_med//2:129+self.num_frames_med], axis=0)
        #img_lg = np.mean(img[128-self.num_frames_lg//2:129+self.num_frames_lg], axis=0)

        return np.stack([img]*3, axis=2)


    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        
        # left_name = self.left_filenames[index].split('/')[1]
        # if left_name.startswith('image'):
        #     self.datapath = self.datapath_15
        # else:
        #     self.datapath = self.datapath_12

        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        
        h, w, _ = left_img.shape
        crop_w, crop_h = 480, 256

        x1 = random.randint(0, w - crop_w)
        if  random.randint(0, 10) >= int(8):
            y1 = random.randint(0, h - crop_h)
        else:
            y1 = random.randint(int(0.3 * h), h - crop_h)

        # random crop
        left_img = left_img[y1:y1 + crop_h, x1:x1 + crop_w]
        right_img = right_img[y1:y1 + crop_h, x1:x1 + crop_w]
        # left_img_med = left_img_med[y1:y1 + crop_h, x1:x1 + crop_w]
        # right_img_med = right_img_med[y1:y1 + crop_h, x1:x1 + crop_w]
        # left_img_lg = left_img_lg[y1:y1 + crop_h, x1:x1 + crop_w]
        # right_img_lg = right_img_lg[y1:y1 + crop_h, x1:x1 + crop_w]
        disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

        # to tensor, normalize
        processed = get_transform()
        left_img = processed(left_img)
        right_img = processed(right_img)
        # left_img_med = processed(left_img_med)
        # right_img_med = processed(right_img_med)
        # left_img_lg = processed(left_img_lg)
        # right_img_lg = processed(right_img_lg)

        return {"left": left_img,
                "right": right_img,
                "disparity": disparity}

        
            # w, h = left_img.size

            # # normalize
            # processed = get_transform()
            # left_img = processed(left_img).numpy()
            # right_img = processed(right_img).numpy()

            # # pad to size 1248x384
            # top_pad = 384 - h
            # right_pad = 1248 - w
            # assert top_pad > 0 and right_pad > 0
            # # pad images
            # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
            #                        constant_values=0)
            # # pad disparity gt
            # if disparity is not None:
            #     assert len(disparity.shape) == 2
            #     disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)


            # if disparity is not None:
            #     return {"left": left_img,
            #             "right": right_img,
            #             "disparity": disparity,
            #             "top_pad": top_pad,
            #             "right_pad": right_pad}
            # else:
            #     return {"left": left_img,
            #             "right": right_img,
            #             "top_pad": top_pad,
            #             "right_pad": right_pad,
            #             "left_filename": self.left_filenames[index],
            #             "right_filename": self.right_filenames[index]}