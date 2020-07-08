r"""PF-PASCAL dataset"""
import os

import scipy.io as sio
import pandas as pd
import numpy as np
import torch

from .dataset import CorrespondenceDataset
from PIL import Image


class PFPascalDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split, cam):
        r"""PF-PASCAL dataset constructor"""
        super(PFPascalDataset, self).__init__(benchmark, datapath, thres, device, split)

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1
        self.cam = cam

        if split == 'trn':
            self.flip = self.train_data.iloc[:, 3].values.astype('int')
        self.src_kps = []
        self.trg_kps = []
        self.src_bbox = []
        self.trg_bbox = []
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                if len(torch.isnan(src_kk).nonzero()) != 0 or \
                        len(torch.isnan(trg_kk).nonzero()) != 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t())
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

    def __getitem__(self, idx):
        r"""Construct and return a batch for PF-PASCAL dataset"""
        sample = super(PFPascalDataset, self).__getitem__(idx)

        sample['src_bbox'] = self.src_bbox[idx].to(self.device)
        sample['trg_bbox'] = self.trg_bbox[idx].to(self.device)
        sample['pckthres'] = self.get_pckthres(sample).to(self.device)

        # Horizontal flip of key-points when training (no training in HyperpixelFlow)
        if self.split == 'trn' and self.flip[idx]:
            sample['src_kps'][0] = sample['src_img'].size()[2] - sample['src_kps'][0]
            sample['trg_kps'][0] = sample['trg_img'].size()[2] - sample['trg_kps'][0]

        src_mask = self.get_mask(self.src_imnames, idx)
        trg_mask = self.get_mask(self.trg_imnames, idx)
        if src_mask is not None and trg_mask is not None:
            sample['src_mask'] = src_mask
            sample['trg_mask'] = trg_mask

        return sample

    def get_image(self, img_names, idx):
        r"""Return image tensor"""
        img_name = os.path.join(self.img_path, img_names[idx])
        image = self.get_imarr(img_name)

        # Data augmentation: horizontal flip (no training in HyperpixelFlow)
        if self.split == 'trn' and self.flip[idx]:
            image = np.flip(image, 1)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image

    def get_mask(self, img_names, idx):
        r"""Return image mask"""
        img_name = os.path.join(self.img_path, img_names[idx])
        mask_name = img_name.replace('/JPEGImages', '-'+self.cam)
        if os.path.exists(mask_name):
            mask = np.array(Image.open(mask_name)) # WxH
        else:
            #print(img_name,mask_name)
            mask = None
        
        return mask

    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        return super(PFPascalDataset, self).get_pckthres(sample)

    def get_points(self, pts, idx):
        r"""Return key-points of an image"""
        return super(PFPascalDataset, self).get_points(pts, idx)


def read_mat(path, obj_name):
    r"""Read specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj
