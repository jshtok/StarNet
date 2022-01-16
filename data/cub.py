from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchnet as tnt

import h5py
from PIL import Image
from PIL import ImageEnhance

#DATASET_DIR = '/dccstor/moshel/datasets/CUB'
DATASET_DIR = '/dccstor/jsdata1/data/starnet_data/CUB'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


class Cub(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False):

        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        print('Loading CUB dataset - phase {0}'.format(phase))
        file_train = os.path.join(DATASET_DIR, 'CUB_train.pickle')
        file_val = os.path.join(DATASET_DIR, 'CUB_val.pickle')
        file_test = os.path.join(DATASET_DIR, 'CUB_test.pickle')

        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

        elif self.phase == 'val' or self.phase == 'test':
            if self.phase == 'test':
                data_novel = load_data(file_test)
            else:  # phase=='val'
                data_novel = load_data(file_val)
            self.data = data_novel['data']
            self.labels = data_novel['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase == 'test' or self.phase == 'val') or (do_not_use_random_transf == True):
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class FewShotDataloader:
    """
    K way - number of classes.
    N shot - number of training examples per class.
    Note: random.sample samples elements without replacement.
    """
    def __init__(self,
                 dataset,
                 nKnovel=5,  # number of novel categories.
                 nKbase=-1,  # number of base categories.
                 nExemplars=1,  # number of training examples per novel category.
                 nTestNovel=15 * 5,  # number of test examples for all the novel categories.
                 nTestBase=15 * 5,  # number of test examples for all the base categories.
                 batch_size=1,  # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000,  # number of batches per epoch.
                 ):
        self.dataset = dataset
        self.phase = self.dataset.phase
        self.num_ways = nKnovel
        self.num_reps = nExemplars
        self.num_quer = nTestNovel
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

    def sample_ids_from_cat(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert (cat_id in self.dataset.label2ind)
        assert (len(self.dataset.label2ind[cat_id]) >= sample_size)
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sample_categories(self, num_cats):
        labelIds = self.dataset.labelIds
        return sorted(random.sample(labelIds, num_cats))

    def sample_reps_and_quer_ids(self, cats_ids, num_quer, num_reps):
        """Samples train and test examples of the novel categories.

        Args:
            cats_ids: a list with the ids of the novel categories.
            num_quer: the total number of test images that will be sampled
                from all the novel categories.
            num_reps: the number of training examples per novel category that
                will be sampled.

        Returns:
            quer_ids: a list of length `num_quer` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(cats_ids) - 1]).
            reps_ids: a list of length len(cats_ids) * num_reps of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(cats_ids) - 1]).
        """
        num_ways = len(cats_ids)
        quer_ids = []
        reps_ids = []
        assert ((num_quer % num_ways) == 0)
        num_quer_per_cat = int(num_quer / num_ways)

        for cat_id in range(len(cats_ids)):
            im_ids = self.sample_ids_from_cat(cats_ids[cat_id], sample_size=(num_quer_per_cat + num_reps))
            im_ids_quer = im_ids[:num_quer_per_cat]
            im_ids_reps = im_ids[num_quer_per_cat:]

            quer_ids += [(img_id, cat_id) for img_id in im_ids_quer]
            reps_ids += [(img_id, cat_id) for img_id in im_ids_reps]
        assert (len(quer_ids) == num_quer)
        assert (len(reps_ids) == len(cats_ids) * num_reps)
        random.shuffle(reps_ids)
        return quer_ids, reps_ids

    def sample_episode(self):
        cats = self.sample_categories(self.num_ways)
        quer_ids, reps_ids = self.sample_reps_and_quer_ids(cats, self.num_quer, self.num_reps)
        random.shuffle(quer_ids)
        return reps_ids, quer_ids, cats

    def ids_to_tensors(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            reps_ids, quer_ids, cats = self.sample_episode()
            Xt, Yt = self.ids_to_tensors(quer_ids)
            Xe, Ye = self.ids_to_tensors(reps_ids)
            return Xe, Ye, Xt, Yt, 0, 0

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
