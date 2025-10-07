# @Time    : 2025/10/7
# @Author  : OpenStereo
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres


class UWStereoDataset(DatasetTemplate):
    """
    UWStereo Dataset Loader
    
    Directory structure:
    UWScene
    └───coral
    |   └───images
    |   |    └───left
    |   |        └───*.png
    |   |    └───right
    |   |        └───*.png
    |   └───disparity
    |        └───*.pfm
    └───coral_reef
    |   └───images
    |   ...
    """
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        if hasattr(self.data_info, 'RETURN_SUPER_PIXEL'):
            self.retrun_super_pixel = self.data_info.RETURN_SUPER_PIXEL
        else:
            self.retrun_super_pixel = False

        if hasattr(self.data_info, 'RETURN_POS'):
            self.retrun_pos = self.data_info.RETURN_POS
        else:
            self.retrun_pos = False

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        
        # Load images
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        
        # Load disparity map
        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        # Handle invalid disparity values
        disp_img[disp_img == np.inf] = 0
        disp_img = np.nan_to_num(disp_img, nan=0.0)
        
        # Initialize occlusion mask (all non-occluded by default)
        occ_mask = np.zeros_like(disp_img, dtype=bool)
        
        sample = {
            'left': left_img,      # [H, W, 3]
            'right': right_img,    # [H, W, 3]
            'disp': disp_img,      # [H, W]
            'occ_mask': occ_mask   # [H, W]
        }

        # Optional: Add superpixel labels
        if self.retrun_super_pixel and self.mode == 'training':
            save_name = Path(left_img_path).relative_to(self.root)
            super_pixel_label = Path(self.root).parent.joinpath(
                'SuperPixelLabel/UWStereo', save_name)
            super_pixel_label = str(super_pixel_label)[:-len('.png')] + "_lsc_lbl.png"
            
            if not os.path.exists(os.path.dirname(super_pixel_label)):
                os.makedirs(os.path.dirname(super_pixel_label), exist_ok=True)
            
            if not os.path.exists(super_pixel_label):
                img = cv2.cvtColor(left_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                cv2.imwrite(super_pixel_label, label.astype(np.uint16))
            
            super_pixel_label = cv2.imread(super_pixel_label, 
                                          cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if super_pixel_label is None:
                img = cv2.cvtColor(left_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                super_pixel_label = label.astype(np.int32)
            else:
                super_pixel_label = super_pixel_label.astype(np.int32)
            
            sample['super_pixel_label'] = super_pixel_label

        # Optional: Add position encoding
        if self.retrun_pos and self.mode == 'training':
            sample['pos'] = get_pos_fullres(1400, sample['left'].shape[1], 
                                           sample['left'].shape[0])

        # Apply transformations
        sample = self.transform(sample)
        
        # Add valid mask and metadata
        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_img_path

        return sample
