r"""Superclass for semantic correspondence datasets"""
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import numpy as np


class CorrespondenceDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, Caltech, and SPair"""
    def __init__(self, benchmark, datapath, thres, device, split):
        r"""CorrespondenceDataset constructor"""
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('PF-WILLOW',
                         'test_pairs.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': ('PF-PASCAL',
                         '_pairs.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'caltech':  ('Caltech-101',
                         'test_pairs_caltech_with_category.csv',
                         '101_ObjectCategories',
                         '',
                         ''),
            'spair':   ('SPair-71k',
                        'Layout/large',
                        'JPEGImages',
                        'PairAnnotation',
                        'bbox')
        }

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres
        self.transform = Normalize(['src_img', 'trg_img'])
        self.device = device
        self.split = split

        # To get initialized in subclass constructors
        self.src_imnames = []
        self.trg_imnames = []
        self.train_data = []
        self.src_kps = []
        self.trg_kps = []
        self.cls_ids = []
        self.cls = []

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        r"""Construct and return a batch"""

        # Image names
        sample = dict()
        sample['src_imname'] = self.src_imnames[idx]
        sample['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        sample['pair_classid'] = self.cls_ids[idx]
        sample['pair_class'] = self.cls[sample['pair_classid']]

        # Image tensors
        sample['src_img'] = self.get_image(self.src_imnames, idx)
        sample['trg_img'] = self.get_image(self.trg_imnames, idx)

        # Key-points
        sample['src_kps'] = self.get_points(self.src_kps, idx).to(self.device)
        sample['trg_kps'] = self.get_points(self.trg_kps, idx).to(self.device)

        # The number of pairs in training split
        sample['datalen'] = len(self.train_data)

        # Transform source & target images (normalization)
        if self.transform:
            sample = self.transform(sample)
        sample['src_img'] = sample['src_img'].to(self.device)
        sample['trg_img'] = sample['trg_img'].to(self.device)

        return sample

    def get_image(self, img_names, idx):
        r"""Return image tensor"""
        img_name = os.path.join(self.img_path, img_names[idx])
        image = self.get_imarr(img_name)
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return image

    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        if self.thres == 'bbox':
            trg_bbox = sample['trg_bbox']
            return torch.max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1])
        elif self.thres == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts, idx):
        r"""Return key-points of an image"""
        return pts[idx]

    def get_imarr(self, path):
        r"""Read a single image file as numpy array from path"""
        return np.array(Image.open(path).convert('RGB'))


class UnNormalize:
    r"""Image unnormalization"""
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image):
        img = image.clone()
        for im_channel, mean, std in zip(img, self.mean, self.std):
            im_channel.mul_(std).add_(mean)
        return img


class Normalize:
    r"""Image normalization"""
    def __init__(self, image_keys, norm_range=True):
        self.image_keys = image_keys
        self.norm_range = norm_range
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        for key in self.image_keys:
            if self.norm_range:
                sample[key] /= 255.0
            sample[key] = self.normalize(sample[key])
        return sample
