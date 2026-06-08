import numpy as np
import glob
import logging
import torch
import random
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from PIL import Image


"""
1、将四个列表（lst0, lst1, lst2, lst3）的元素按对应位置组合成一个新的列表（lst）
2、对这个新列表进行随机打乱（shuffle）
3、将打乱后的列表重新拆分成四个列表，分别赋值回lst0, lst1, lst2, lst3
"""
def syn_shuffle(lst0, lst1, lst2, lst3):
    lst = list(zip(lst0, lst1, lst2, lst3))
    random.shuffle(lst)
    lst0, lst1, lst2, lst3 = zip(*lst)
    return lst0, lst1, lst2, lst3


class PipelineDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase, category, split_ratio=0.8):
        self.phase = phase
        if self.phase in ('train', 'eval'):
            self.img_path = os.path.join(root, category, 'train')
        else:
            self.img_path = os.path.join(root, category, 'test')
            self.gt_path = os.path.join(root, category, 'ground_truth')
        self.spit_ratio = split_ratio
        self.transform = transform
        self.gt_transform = gt_transform
        assert os.path.isdir(os.path.join(root, category)), 'Error PipelineDataset category:{}'.format(category)
        # self.labels => good : 0, anomaly : 1
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_paths_list = []
        gt_paths_list = []
        labels_list = []
        types_list = []

        defect_types = os.listdir(self.img_path)
        logging.info("List Image_Path:", defect_types)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_paths_list.extend(img_paths)
                gt_paths_list.extend([0]*len(img_paths))
                labels_list.extend([0]*len(img_paths))
                types_list.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.bmp")

                img_paths.sort()
                gt_paths.sort()
                img_paths_list.extend(img_paths)
                if len(gt_paths) == 0:
                    gt_paths = [0]*len(img_paths)
                gt_paths_list.extend(gt_paths)
                labels_list.extend([1]*len(img_paths))
                types_list.extend([defect_type]*len(img_paths))
        train_len = int(len(img_paths_list)*self.spit_ratio)

        img_paths_list, gt_paths_list, labels_list, types_list = syn_shuffle(img_paths_list, gt_paths_list,
                                                                             labels_list, types_list)
        if self.phase == 'train':
            img_paths_list = img_paths_list[:train_len]
            gt_paths_list = gt_paths_list[:train_len]
            labels_list = labels_list[:train_len]
            types_list = types_list[:train_len]
        elif self.phase == 'eval':
            img_paths_list = img_paths_list[train_len:]
            gt_paths_list = gt_paths_list[train_len:]
            labels_list = labels_list[train_len:]
            types_list = types_list[train_len:]

        return img_paths_list, gt_paths_list, labels_list, types_list

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        origin = img
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return {'origin': np.array(origin), 'image': img, 'gt': gt, 'label': label,
                'name': os.path.basename(img_path[:-4]), 'type': img_type}


class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.imagenet_dir = root
        self.transform = transform
        self.dataset = ImageFolder(self.imagenet_dir, transform=self.transform)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.dataset[idx][0]


def load_infinite(loader):
    # 创建一个迭代器
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)

