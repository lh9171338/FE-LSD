import os
import json
import cv2
import numpy as np
import copy
import torch
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import default_collate


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, event, ann=None):
        for transform in self.transforms:
            image, event, ann = transform(image, event, ann)
        return image, event, ann


class Resize(object):
    def __init__(self, image_size, ann_size):
        self.image_height = image_size[1]
        self.image_width = image_size[0]
        self.ann_height = ann_size[1]
        self.ann_width = ann_size[0]

    def __call__(self, image, event, ann=None):
        image = F.resize(image, (self.image_height, self.image_width))
        if event is not None:
            event = cv2.resize(event, (self.image_width, self.image_height), cv2.INTER_LINEAR)
        if ann is None:
            return image, event
        else:
            sx = self.ann_width / ann['width']
            sy = self.ann_height / ann['height']
            ann['junc'][:, 0] = np.clip(ann['junc'][:, 0] * sx, 0, self.ann_width - 1e-4)
            ann['junc'][:, 1] = np.clip(ann['junc'][:, 1] * sy, 0, self.ann_height - 1e-4)
            return image, event, ann


class ResizeImage(object):
    def __init__(self, image_size):
        self.image_height = image_size[1]
        self.image_width = image_size[0]

    def __call__(self, image, event, ann=None):
        image = F.resize(image, (self.image_height, self.image_width))
        if event is not None:
            event = cv2.resize(event, (self.image_width, self.image_height), cv2.INTER_LINEAR)
        if ann is None:
            return image, event
        else:
            return image, event, ann


class ToTensor(object):
    def __call__(self, image, event, ann=None):
        image = F.to_tensor(image)
        if event is not None:
            event = torch.from_numpy(event).permute(2, 0, 1)
        if ann is None:
            return image, event
        else:
            for key, val in ann.items():
                if isinstance(val, np.ndarray):
                    ann[key] = torch.from_numpy(val)
            return image, event, ann


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device)
        image = image.mul(std[:, None, None]).add(mean[:, None, None])
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, event, ann=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if ann is None:
            return image, event
        else:
            return image, event, ann


class Dataset(Data.Dataset):
    """
    Format of the annotation file
    annotations[i] has the following dict items:
    - filename  # of the input image, str
    - height    # of the input image, int
    - width     # of the input image, int
    - lines     # of the input image, list of list, N*4
    - junc      # of the input image, list of list, M*2
    """

    def __init__(self, cfg, split):
        self.split = split
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.heatmap_size
        self.mean = cfg.mean
        self.std = cfg.std

        root = cfg.dataset_path
        self.ann_file = os.path.join(root, f'{split}.json')
        with open(self.ann_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_file_list = [os.path.join(root, 'images-blur', self.annotations[i]['filename'])
                                for i in range(len(self.annotations))]
        self.event_file_list = [os.path.join(root, 'events', self.annotations[i]['filename'].replace('.png', '.npz'))
                                for i in range(len(self.annotations))]

    def __getitem__(self, index):
        if self.split == 'train':
            reminder = index // len(self.annotations)
            index = index % len(self.annotations)
            ann = copy.deepcopy(self.annotations[index])
            for key, type in (['junc', np.float32],
                              ['edges_positive', np.long],
                              ['edges_negative', np.long]):
                ann[key] = np.array(ann[key], dtype=type)
            width = ann['width']
            height = ann['height']

            image = np.asarray(Image.open(self.image_file_list[index]).convert('RGB'))
            with np.load(self.event_file_list[index]) as events:
                event = events['event'].astype(np.float32)

            if reminder == 1:
                image = image[:, ::-1, :]
                event = event[:, ::-1, :]
                ann['junc'][:, 0] = width - ann['junc'][:, 0]
            elif reminder == 2:
                image = image[::-1, :, :]
                event = event[::-1, :, :]
                ann['junc'][:, 1] = height - ann['junc'][:, 1]
            elif reminder == 3:
                image = image[::-1, ::-1, :]
                event = event[::-1, ::-1, :]
                ann['junc'][:, 0] = width - ann['junc'][:, 0]
                ann['junc'][:, 1] = height - ann['junc'][:, 1]
            image = Image.fromarray(image)
        else:
            ann = copy.deepcopy(self.annotations[index])
            image = Image.open(self.image_file_list[index]).convert('RGB')
            with np.load(self.event_file_list[index]) as events:
                event = events['event'].astype(np.float32)

        image, event, ann = self.transforms(image, event, ann)
        fusion = torch.cat((image, event))
        return fusion, ann

    def __len__(self):
        if self.split == 'train':
            return len(self.annotations) * 4
        else:
            return len(self.annotations)

    def transforms(self, image, event, ann=None):
        if self.split == 'train':
            transforms = Compose([
                Resize(self.image_size, self.heatmap_size),
                ToTensor(),
                Normalize(self.mean, self.std)
            ])
        else:
            transforms = Compose([
                ResizeImage(self.image_size),
                ToTensor(),
                Normalize(self.mean, self.std)
            ])

        image, event, ann = transforms(image, event, ann)
        return image, event, ann

    @staticmethod
    def collate(batch):
        return default_collate([b[0] for b in batch]), [b[1] for b in batch]
