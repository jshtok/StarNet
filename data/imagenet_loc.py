# Dataloader of Gidaris & Komodakis, CVPR 2018
# Adapted from:
# https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py
# adapted for StarNet by Amit Alfasy and Joseph Shtok, IBM Research AI, Oct. 2019
#------------------------------------------------------------------------------------------
from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import json
import math
import PIL.Image as PILI

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
from torchvision.datasets.folder import ImageFolder,DatasetFolder
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchnet as tnt

import h5py

from PIL import Image
from PIL import ImageEnhance

from pdb import set_trace as breakpoint
from aux_routines.show_boxes import show_gt_boxes
from utils import assert_folder
imagenet_loc_data_path = '/dccstor/leonidka1/data/imagenet/ILSVRC/Data/CLS-LOC/'
imagenet_loc_anno_path = "/dccstor/leonidka1/data/imagenet/ILSVRC/Annotations/CLS-LOC/"
# Set the appropriate paths of the datasets here.

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

def dummy_loader(item):
    return item

def load_imagenet_annotation(xml_filename,img_filename,cls):
    """
    for a given index, load image and bounding boxes info from XML file
    :param index: index of a specific image
    :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    """
    import xml.etree.ElementTree as ET
    num_classes =1000
    roi_rec = dict()
    if xml_filename.endswith('.JPEG'):
        xml_filename = xml_filename.replace('/Data/', '/Annotations/').replace('.JPEG', '.xml')
    if not os.path.isfile(xml_filename):
        print(f'file {xml_filename} not found')
        raise FileNotFoundError
    else:
        tree = ET.parse(xml_filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        if True: # not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if (obj.find('difficult') is None) or int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        #overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
        from PIL import Image
        # im = Image.open(img_filename)
        im = Image.open(img_filename).convert('RGB')
        # class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = np.maximum(float(bbox.find('xmin').text) - 1,0)
            y1 = np.maximum(float(bbox.find('ymin').text) - 1,0)
            x2 = np.maximum(float(bbox.find('xmax').text) - 1,0)
            y2 = np.maximum(float(bbox.find('ymax').text) - 1,0)

            if (x2<=x1) or (y2<=y1):
                assert (x2<=x1) or (y2<=y1), 'tlc>brc'


            width, height = im.size
            if (roi_rec['width'] != width) or (roi_rec['height'] != height):
                assert (roi_rec['width'] != width) or (roi_rec['height'] != height), 'wrongly recorded image size'

            if (y2>=height) or (x2>=width):
                assert (y2>=height) or (x2>=width), 'bb exceeds image boundaries'

            #cls = self._wnid_to_ind[str(obj.find("name").text).lower().strip()]
            # cls = class_to_index[obj.find('name').text.lower().strip()]

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            #overlaps[ix, cls] = 1.0

    roi_rec.update({'boxes': boxes,
                    'gt_classes': gt_classes,
                    'flipped': False})
    return roi_rec,im

#  torch.utils.data.Dataset
class AnnoImageNet(DatasetFolder):
    def __init__(self, options, phase='test', load_data_from_file=False, do_not_use_random_transf=False, debug=False):
        anno_path = os.path.join(imagenet_loc_anno_path, 'train')
        # current Imagenet-LOC split:
        self.training_classes_list_path = "data/inloc_first101_categories_sn.txt"
        self.test_classes_list_path = "data/in_domain_categories_sn.txt"

        if hasattr(options,'test_classes_list_path') and not options.test_classes_list_path=='None':
            self.test_classes_list_path = options.test_classes_list_path

        super(AnnoImageNet, self).__init__(anno_path, dummy_loader, extensions=('.xml'))
        idx_to_class = {}
        for key in self.class_to_idx.keys():
            idx_to_class[self.class_to_idx[key]] = key
        self.idx_to_class = idx_to_class
        # self.prepare_data_and_labels_val()
        # self.create_ImageNet_Train_val_data()
        # self.create_ImageNet_test_data()
        # self.base_folder = 'miniImagenet'
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'ImageNet_' + phase
        self.load_data_from_file = load_data_from_file
        self.image_res = options.image_res
        assert((options.pad_mode == 'constant') or (options.pad_mode == 'edge') or (options.pad_mode == 'reflect') or
           (options.pad_mode == 'symmetric'))

        self.pad_mode = options.pad_mode
        self.debug = debug
        self.mean_pix = [0.485, 0.456, 0.406]
        self.std_pix = [0.229, 0.224, 0.225]
        if hasattr(options,'crop_style'):
            self.crop_style = options.crop_style
        else:
            self.crop_style = 1

        if options.recompute_dataset_dicts==1:
            if phase == 'train' or phase == 'val':
                self.create_data_dictionaries_train_val()
            if phase == 'test':
                self.create_data_dictionaries_test()

        print('Loading ImageNet dataset - phase {0}'.format(phase))

        idx_to_class = {}
        for key in self.class_to_idx.keys():
            idx_to_class[self.class_to_idx[key]] = key
        self.idx_to_class = idx_to_class
        # self.prepare_data_and_labels_val()
        # self.create_ImageNet_Train_val_data()
        # self.create_ImageNet_test_data()
        # self.base_folder = 'miniImagenet'
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'ImageNet_' + phase
        self.load_data_from_file = load_data_from_file
        self.image_res = options.image_res
        assert((options.pad_mode == 'constant') or (options.pad_mode == 'edge') or (options.pad_mode == 'reflect') or
           (options.pad_mode == 'symmetric'))

        self.pad_mode = options.pad_mode
        self.debug = debug
        self.mean_pix = [0.485, 0.456, 0.406]
        self.std_pix = [0.229, 0.224, 0.225]
        if hasattr(options,'crop_style'):
            self.crop_style = options.crop_style
        else:
            self.crop_style = 1

        if options.recompute_dataset_dicts==1:
            if phase == 'train' or phase == 'val':
                self.create_data_dictionaries_train_val()
            if phase == 'test':
                self.create_data_dictionaries_test()

        print('Loading ImageNet dataset - phase {0}'.format(phase))
        # file_train_categories_train_phase_data = '/dccstor/girassy/StarNet/data/ImageNetTrainCatTrainPhaseData.pkl.npy'
        # file_train_categories_train_phase_labels = '/dccstor/girassy/StarNet/data/ImageNetTrainCatTrainPhaseLabels.pkl'
        # file_train_categories_val_phase_data = '/dccstor/girassy/StarNet/data/ImageNetTrainCatValPhaseData.pkl.npy'
        # file_train_categories_val_phase_labels = '/dccstor/girassy/StarNet/data/ImageNetTrainCatValPhaseLabels.pkl'
        # file_train_categories_test_phase_data = '/dccstor/girassy/StarNet/data/ImageNetTrainCatTestPhaseData.pkl'
        # file_train_categories_test_phase_labels = '/dccstor/girassy/StarNet/data/ImageNetTrainCatTestPhaseLabels.pkl'
        # file_val_categories_val_phase_data = '/dccstor/girassy/StarNet/data/ImageNetValData.pkl.npy'
        # file_val_categories_val_phase_labels = '/dccstor/girassy/StarNet/data/ImageNetValLabels.pkl'
        # file_test_categories_test_phase_data = '/dccstor/girassy/StarNet/data/ImageNetTestCatTestPhaseData.pkl'
        # file_test_categories_test_phase_labels = '/dccstor/girassy/StarNet/data/ImageNetTestCatTestPhaseLabels.pkl'

        # data_folder = os.path.join(data_folder, options.dataset)
        # os.makedirs(data_folder,exist_ok=True)
        # file_train_categories_train_phase_data = os.path.join(data_folder,'ImageNetTrainCatTrainPhaseData.pkl.npy')
        # file_train_categories_train_phase_labels = os.path.join(data_folder,'ImageNetTrainCatTrainPhaseLabels.pkl')
        # file_train_categories_val_phase_data = os.path.join(data_folder,'ImageNetTrainCatValPhaseData.pkl.npy')
        # file_train_categories_val_phase_labels = os.path.join(data_folder,'ImageNetTrainCatValPhaseLabels.pkl')
        # file_train_categories_test_phase_data = os.path.join(data_folder,'ImageNetTrainCatTestPhaseData.pkl')
        # file_train_categories_test_phase_labels = os.path.join(data_folder,'ImageNetTrainCatTestPhaseLabels.pkl')
        # file_val_categories_val_phase_data = os.path.join(data_folder,'ImageNetValData.pkl.npy')
        # file_val_categories_val_phase_labels = os.path.join(data_folder,'ImageNetValLabels.pkl')
        # file_test_categories_test_phase_data = os.path.join(data_folder,'ImageNetTestCatTestPhaseData.pkl')
        # file_test_categories_test_phase_labels = os.path.join(data_folder,'ImageNetTestCatTestPhaseLabels.pkl')

        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            if load_data_from_file:
                self.data = np.load(file_train_categories_train_phase_data)
                self.labels = load_data(file_train_categories_train_phase_labels)
            else:
                self.prepare_data_and_labels_train()
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase == 'val' or self.phase == 'test':
            if self.phase == 'test':
                if load_data_from_file:
                    # load data that will be used for evaluating the recognition
                    # accuracy of the base categories.
                    data_base = load_data(file_train_categories_test_phase_data)
                    labels_base = load_data(file_train_categories_test_phase_labels)
                    # load data that will be use for evaluating the few-shot recogniton
                    # accuracy on the novel categories.
                    data_novel = load_data(file_test_categories_test_phase_data)
                    labels_novel = load_data(file_test_categories_test_phase_labels)
                else:
                    data_base, labels_base, data_novel, labels_novel = self.prepare_data_and_labels_test()
            else:  # phase=='val'
                if load_data_from_file:
                    # load data that will be     used for evaluating the recognition
                    # accuracy of the base categories.
                    data_base = np.load(file_train_categories_val_phase_data)
                    labels_base = load_data(file_train_categories_val_phase_labels)
                    # load data that will be use for evaluating the few-shot recogniton
                    # accuracy on the novel categories.
                    data_novel = np.load(file_val_categories_val_phase_data)
                    labels_novel = load_data(file_val_categories_val_phase_labels)
                else:
                    data_base, labels_base, data_novel, labels_novel = self. prepare_data_and_labels_val()

            self.data = np.concatenate(
                [data_base, data_novel], axis=0)
            self.labels = labels_base + labels_novel
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = buildLabelIndex(labels_base).keys()
            self.labelIds_novel = buildLabelIndex(labels_novel).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert (len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        # mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        # std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

        normalize = transforms.Normalize(mean=self.mean_pix, std=self.std_pix)

        if (self.phase == 'test' or self.phase == 'val') or (do_not_use_random_transf == True):
            if load_data_from_file:
                self.transform = transforms.Compose([
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            if load_data_from_file:
                self.transform = transforms.Compose([
                    #transforms.RandomCrop(self.image_res, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    #transforms.RandomCrop(self.image_res, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])

        # roidb_ni_fname = '/dccstor/jsdata1/data/inloc_roidb_ni_s.pkl'
        # with open(roidb_ni_fname, 'rb')as fid:
        #     self.roidb_ni = pickle.load(fid)

    def __getitem__(self, index):
        xml, label = self.data[index], self.labels[index]

        if self.phase == 'test':
            img = xml.replace('.xml','.JPEG').replace('Annotations/CLS-LOC','Data/CLS-LOC')

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            gt_entry, img = load_imagenet_annotation(xml, img, label)
            # show_gt_boxes(img,gt_entry['boxes'],[str(i) for i in gt_entry['gt_classes']],save_file_path='/dccstor/jsdata1/dev/tmp.jpg')
            # DB_entry['gt_classes'],DB_entry['boxes'],DB_entry['gt_names']
        else:
            from PIL import Image
            # im = Image.open(img_filename)
            img = Image.open(xml).convert('RGB')

            gt_entry = None

        old_size = img.size #[columns, rows]
        if self.crop_style==1:
            larger = 0 if img.size[0] > img.size[1] else 1
            new_size = int((self.image_res * img.size[abs(larger - 1)]) / img.size[larger])
            resize_transform = transforms.Resize(new_size)
            img = resize_transform(img)
            if gt_entry is not None:
                gt_entry['boxes'] = gt_entry['boxes'].astype('float')
                gt_entry['boxes']*=float(new_size)/float(min(old_size))
                gt_entry['boxes'] = gt_entry['boxes'].round().astype('int16')
            pad_width_left = int((self.image_res - img.size[0]) / 2)
            pad_width_right = int(pad_width_left + ((self.image_res - img.size[0]) % 2))
            pad_width_top = int((self.image_res - img.size[1]) / 2)
            pad_width_bottom = int(pad_width_top + ((self.image_res - img.size[1]) % 2))
            pad_transform = transforms.Pad((pad_width_left, pad_width_top, pad_width_right, pad_width_bottom),
                                           padding_mode=self.pad_mode)
            img = pad_transform(img)
            if gt_entry is not None:
                gt_entry['boxes'][:, [0, 2]]+= pad_width_left
                gt_entry['boxes'][:, [1, 3]] += pad_width_top
        elif self.crop_style==2:
            downsize_tfm = transforms.Compose([
                transforms.Resize(self.image_res),  # resize to have the smaller dimension equal to image_res
                transforms.CenterCrop(self.image_res) # crop the other dimension to image_res
            ])
            img = downsize_tfm(img)
            img_scale = float(self.image_res)/float(min(old_size))
            if gt_entry is not None:
                gt_entry['boxes'] = gt_entry['boxes'].astype('float')*img_scale
            if old_size[1]>old_size[0]: #rows>columns, portrait. crop top-bottom. img_size is now [img_scale*old_size[1],image_res]
                crop_height= int((img_scale*old_size[1] - self.image_res) / 2)
                if gt_entry is not None:
                    gt_entry['boxes'][:, [1, 3]] -= crop_height
            else: # crop left-right
                crop_width = int((img_scale*old_size[0] -self.image_res) / 2)
                if gt_entry is not None:
                    gt_entry['boxes'][:, [0, 2]] -= crop_width
            if gt_entry is not None:
                gt_entry['boxes'] = gt_entry['boxes'].round().astype('int16')
        else:
            os.error('Imagenet_loc __getitem__: Unrecognized crop stype')

        #show_gt_boxes(img, gt_entry['boxes'], [str(i) for i in gt_entry['gt_classes']], save_file_path='/dccstor/jsdata1/dev/tmp.jpg')
        if self.transform is not None:
            img = self.transform(img)

        if gt_entry is not None:
            boxes = np.pad(gt_entry['boxes'],((0,10-gt_entry['boxes'].shape[0]),(0,0)),'constant')
        else:
            boxes = None

        if self.phase == 'test':
            return img, label, torch.tensor(boxes)
        else:
            return img, label

    def __len__(self):
        return len(self.data)

    def prepare_data_and_labels_train(self):
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize])

        # get train classes indices
        ClassIdxDictTrainTrain = load_data('./datasets/ImageNet/ImageNetClassIdxDictTrainCatTrainPhase.pkl',)
        # save train cat train phase data to file
        data_train_train = []
        labels_train_train = []
        for i in ClassIdxDictTrainTrain.keys():
            for j, idx in enumerate(ClassIdxDictTrainTrain[i]):
                path, label = self.samples[idx]
                data_train_train += [path]
                labels_train_train += [label]
                if self.debug and j == 50:
                    break
        print("data train cat train phase shape: {}".format(len(data_train_train)))
        print("labels train cat train phase shape: {}".format(len(labels_train_train)))
        self.data = data_train_train
        self.labels = labels_train_train

    def prepare_data_and_labels_val(self):
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize])

        # get train classes indices
        ClassIdxDictTrainVal = load_data('datasets/ImageNet/ImageNetClassIdxDictTrainCatValPhase.pkl')
        # save train cat train phase data to file
        data_train_val = []
        labels_train_val = []
        for i in ClassIdxDictTrainVal.keys():
            for j, idx in enumerate(ClassIdxDictTrainVal[i]):
                path, label = self.samples[idx]
                data_train_val += [path]
                labels_train_val += [label]
                if self.debug and j == 50:
                    break
        print("data train cat val phase shape: {}".format(len(data_train_val)))
        print("labels train cat val phase shape: {}".format(len(labels_train_val)))
        # get validation classes indices
        val_indices = load_data('data/ImageNetValClasses.pkl')
        ClassIdxDict = load_data('data/ImageNetClassIdxDict.pkl')
        # save val cat val phase data to file
        data_val = []
        labels_val = []
        for i in val_indices:
            for j, idx in enumerate(ClassIdxDict[i]):
                path, label = self.samples[idx]
                data_val += [path]
                labels_val += [label]
                if self.debug and j == 50:
                    break
        print("data val shape: {}".format(len(data_val)))
        print("labels val shape: {}".format(len(labels_val)))

        return data_train_val, labels_train_val, data_val, labels_val

    def prepare_data_and_labels_test(self):
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize])

        # get train classes indices
        base_folder = assert_folder('./datasets/ImageNet')
        train_classes_path = os.path.join(base_folder, 'ImageNetTrainClasses.pkl')
        ClassIdxDict_Test_path = os.path.join(base_folder, 'ImageNetClassIdxDictTest.pkl')
        TestClasses_path = os.path.join(base_folder, 'ImageNetTestClasses.pkl')
        train_indices = load_data(train_classes_path)
        ClassIdxDictTest = load_data(ClassIdxDict_Test_path)
        # save train cat train phase data to file
        data_test_train = []
        labels_test_train = []
        for i in train_indices:
            for j, idx in enumerate(ClassIdxDictTest[i]):
                path, label = self.samples[idx]
                data_test_train += [path]
                labels_test_train += [label]
                if self.debug and j == 50:
                    break
        print("data train cat test phase shape: {}".format(len(data_test_train)))
        print("labels train cat test phase shape: {}".format(len(labels_test_train)))
        # get validation classes indices
        test_indices = load_data(TestClasses_path)
        # save val cat val phase data to file
        data_test_test = []
        labels_test_test = []
        for i in test_indices:
            for j, idx in enumerate(ClassIdxDictTest[i]):
                path, label = self.samples[idx]
                data_test_test += [path]
                labels_test_test += [label]

        print("data val shape: {}".format(len(data_test_test)))
        print("labels val shape: {}".format(len(labels_test_test)))

        return data_test_train, labels_test_train, data_test_test, labels_test_test

    def create_data_dictionaries_train_val(self):
        base_folder = assert_folder('./datasets/ImageNet')
        #val_classes_list_path = os.path.join(base_folder,'imagenet_val_classes.txt')
        train_classes_path = os.path.join(base_folder,'ImageNetTrainClasses.pkl')
        ClassIdxDict_TrainCatTrainPhase_path = os.path.join(base_folder,'ImageNetClassIdxDictTrainCatTrainPhase.pkl')
        ClassIdxDict_TrainCatValPhase_path = os.path.join(base_folder,'ImageNetClassIdxDictTrainCatValPhase.pkl')
        IdxDict_path = os.path.join(base_folder,'ImageNetClassIdxDict.pkl')
        ValClasses_path = os.path.join(base_folder,'ImageNetValClasses.pkl')
        train_classes_joseph = open(self.training_classes_list_path).read().splitlines()
        # get train classes indices
        train_indices = []
        for class_name_joseph in train_classes_joseph:
            for folder_name, class_name in folder_name_to_class_name.items():
                if class_name_joseph == class_name:
                    train_indices += [self.class_to_idx[folder_name]]
        print("train indices num: {}".format(len(train_indices)))
        print("train classes list len: {}".format(len(train_classes_joseph)))
        with open(train_classes_path, 'wb') as f:
            pickle.dump(train_indices, f)
        f.close()
        print(train_indices)
        # Dictionary - Keys: labels(int), Values: list of indices of images tagged as $key
        ClassIdxDict = {el: [] for el in range(1000)}
        for idx, (path, label) in enumerate(self.samples):
            ClassIdxDict[label] += [idx]
        # divide train data between train and val phase
        random.seed(0)
        ClassIdxDictTrainTrain = {el: [] for el in train_indices}
        ClassIdxDictTrainVal = {el: [] for el in train_indices}
        for label in train_indices:
            ClassIdxDictTrainVal[label] = random.sample(ClassIdxDict[label], int(len(ClassIdxDict[label]) * 0.1))
            ClassIdxDictTrainTrain[label] = [idx for idx in ClassIdxDict[label] if
                                             idx not in ClassIdxDictTrainVal[label]]
        with open(ClassIdxDict_TrainCatTrainPhase_path, 'wb') as f:
            pickle.dump(ClassIdxDictTrainTrain, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDictTrainTrain.keys())
        with open(ClassIdxDict_TrainCatValPhase_path, 'wb') as f:
            pickle.dump(ClassIdxDictTrainVal, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDictTrainVal.keys())
        with open(IdxDict_path, 'wb') as f:
            pickle.dump(ClassIdxDict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDict.keys())
        # val_classes_list = open(val_classes_list_path).read().splitlines()
        # val_indices = []
        # for class_name_val in val_classes_list:
        #     for folder_name, class_name in folder_name_to_class_name.items():
        #         if class_name_val == class_name:
        #             val_indices += [self.class_to_idx[folder_name]]
        # print("val indices num: {}".format(len(val_indices)))
        # print("val classes num: {}".format(len(val_classes_list)))
        # with open(ValClasses_path, 'wb') as f:
        #     pickle.dump(val_indices, f)
        # f.close()
        # print(val_indices)

    def create_data_dictionaries_test(self):
        base_folder = assert_folder('./datasets/ImageNet')
        # load train classes indices
        train_classes_path = os.path.join(base_folder, 'ImageNetTrainClasses.pkl')
        ClassIdxDict_Test_path = os.path.join(base_folder, 'ImageNetClassIdxDictTest.pkl')
        TestClasses_path = os.path.join(base_folder,'ImageNetTestClasses.pkl')
        train_indices = load_data(train_classes_path)
        print(train_indices)
        # Dictionary - Keys: labels(int), Values: list of indices of images tagged as $key
        ClassIdxDictTest = {el: [] for el in range(1000)}
        for idx, (path, label) in enumerate(self.samples):
            ClassIdxDictTest[label] += [idx]
        with open(ClassIdxDict_Test_path, 'wb') as f:
            pickle.dump(ClassIdxDictTest, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDictTest.keys())
        test_classes_list = open(self.test_classes_list_path).read().splitlines()
        test_indices = []
        for class_name_test in test_classes_list:
            if class_name_test == "crane":
                test_indices += [134]
            else:
                for folder_name, class_name in folder_name_to_class_name.items():
                    if class_name_test == class_name:
                        test_indices += [self.class_to_idx[folder_name]]
                        break
        print("test indices num: {}".format(len(test_indices)))
        print("test classes num: {}".format(len(test_classes_list)))
        with open(TestClasses_path, 'wb') as f:
            pickle.dump(test_indices, f)
        f.close()
        print(test_indices)


class ImageNet(ImageFolder):
    def __init__(self, options, phase='train', load_data_from_file=False, do_not_use_random_transf=False, debug=False):
        data_path = os.path.join(imagenet_loc_data_path, 'train')
        anno_path = os.path.join(imagenet_loc_anno_path, 'train')
        # current Imagenet-LOC split:
        self.training_classes_list_path = "data/inloc_first101_categories_sn.txt"
        self.test_classes_list_path = "data/in_domain_categories_sn.txt"

        if hasattr(options,'test_classes_list_path') and not options.test_classes_list_path=='None':
            self.test_classes_list_path = options.test_classes_list_path

        if phase == 'test':
            #super(ImageNet, self).__init__(anno_path, dummy_loader, extensions=('.xml'))
            super(ImageNet, self).__init__(data_path, dummy_loader)
        else:
            super(ImageNet, self).__init__(data_path,dummy_loader) #,extensions=('.jpeg','.jpg'))
        idx_to_class = {}
        for key in self.class_to_idx.keys():
            idx_to_class[self.class_to_idx[key]] = key
        self.idx_to_class = idx_to_class
        # self.prepare_data_and_labels_val()
        # self.create_ImageNet_Train_val_data()
        # self.create_ImageNet_test_data()
        # self.base_folder = 'miniImagenet'
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'ImageNet_' + phase
        self.load_data_from_file = load_data_from_file
        self.image_res = options.image_res
        assert((options.pad_mode == 'constant') or (options.pad_mode == 'edge') or (options.pad_mode == 'reflect') or
           (options.pad_mode == 'symmetric'))

        self.pad_mode = options.pad_mode
        self.debug = debug
        self.mean_pix = [0.485, 0.456, 0.406]
        self.std_pix = [0.229, 0.224, 0.225]
        if hasattr(options,'crop_style'):
            self.crop_style = options.crop_style
        else:
            self.crop_style = 1

        if options.recompute_dataset_dicts==1:
            if phase == 'train' or phase == 'val':
                self.create_data_dictionaries_train_val()
            if phase == 'test':
                self.create_data_dictionaries_test()

        print('Loading ImageNet dataset - phase {0}'.format(phase))


        idx_to_class = {}
        for key in self.class_to_idx.keys():
            idx_to_class[self.class_to_idx[key]] = key
        self.idx_to_class = idx_to_class
        # self.prepare_data_and_labels_val()
        # self.create_ImageNet_Train_val_data()
        # self.create_ImageNet_test_data()
        # self.base_folder = 'miniImagenet'
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'ImageNet_' + phase
        self.load_data_from_file = load_data_from_file
        self.image_res = options.image_res
        assert((options.pad_mode == 'constant') or (options.pad_mode == 'edge') or (options.pad_mode == 'reflect') or
           (options.pad_mode == 'symmetric'))

        self.pad_mode = options.pad_mode
        self.debug = debug
        self.mean_pix = [0.485, 0.456, 0.406]
        self.std_pix = [0.229, 0.224, 0.225]
        if hasattr(options,'crop_style'):
            self.crop_style = options.crop_style
        else:
            self.crop_style = 1

        if options.recompute_dataset_dicts==1:
            if phase == 'train' or phase == 'val':
                self.create_data_dictionaries_train_val()
            if phase == 'test':
                self.create_data_dictionaries_test()

        # data_folder = os.path.join(data_folder, options.dataset)
        # os.makedirs(data_folder,exist_ok=True)
        # print('Loading ImageNet dataset - phase {0}'.format(phase))
        # file_train_categories_train_phase_data = os.path.join(data_folder,'ImageNetTrainCatTrainPhaseData.pkl.npy')
        # file_train_categories_train_phase_labels = os.path.join(data_folder,'ImageNetTrainCatTrainPhaseLabels.pkl')
        # file_train_categories_val_phase_data = os.path.join(data_folder,'ImageNetTrainCatValPhaseData.pkl.npy')
        # file_train_categories_val_phase_labels = os.path.join(data_folder,'ImageNetTrainCatValPhaseLabels.pkl')
        # file_train_categories_test_phase_data = os.path.join(data_folder,'ImageNetTrainCatTestPhaseData.pkl')
        # file_train_categories_test_phase_labels = os.path.join(data_folder,'ImageNetTrainCatTestPhaseLabels.pkl')
        # file_val_categories_val_phase_data = os.path.join(data_folder,'ImageNetValData.pkl.npy')
        # file_val_categories_val_phase_labels = os.path.join(data_folder,'ImageNetValLabels.pkl')
        # file_test_categories_test_phase_data = os.path.join(data_folder,'ImageNetTestCatTestPhaseData.pkl')
        # file_test_categories_test_phase_labels = os.path.join(data_folder,'ImageNetTestCatTestPhaseLabels.pkl')

        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            if load_data_from_file:
                self.data = np.load(file_train_categories_train_phase_data)
                self.labels = load_data(file_train_categories_train_phase_labels)
            else:
                self.prepare_data_and_labels_train()
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase == 'val' or self.phase == 'test':
            if self.phase == 'test':
                if load_data_from_file:
                    # load data that will be used for evaluating the recognition
                    # accuracy of the base categories.
                    data_base = load_data(file_train_categories_test_phase_data)
                    labels_base = load_data(file_train_categories_test_phase_labels)
                    # load data that will be use for evaluating the few-shot recogniton
                    # accuracy on the novel categories.
                    data_novel = load_data(file_test_categories_test_phase_data)
                    labels_novel = load_data(file_test_categories_test_phase_labels)
                else:
                    data_base, labels_base, data_novel, labels_novel = self.prepare_data_and_labels_test()
            else:  # phase=='val'
                if load_data_from_file:
                    # load data that will be     used for evaluating the recognition
                    # accuracy of the base categories.
                    data_base = np.load(file_train_categories_val_phase_data)
                    labels_base = load_data(file_train_categories_val_phase_labels)
                    # load data that will be use for evaluating the few-shot recogniton
                    # accuracy on the novel categories.
                    data_novel = np.load(file_val_categories_val_phase_data)
                    labels_novel = load_data(file_val_categories_val_phase_labels)
                else:
                    data_base, labels_base, data_novel, labels_novel = self. prepare_data_and_labels_val()

            self.data = np.concatenate(
                [data_base, data_novel], axis=0)
            self.labels = labels_base + labels_novel
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = buildLabelIndex(labels_base).keys()
            self.labelIds_novel = buildLabelIndex(labels_novel).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert (len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        # mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        # std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

        normalize = transforms.Normalize(mean=self.mean_pix, std=self.std_pix)

        if (self.phase == 'test' or self.phase == 'val') or (do_not_use_random_transf == True):
            if load_data_from_file:
                self.transform = transforms.Compose([
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            if load_data_from_file:
                self.transform = transforms.Compose([
                    #transforms.RandomCrop(self.image_res, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    #transforms.RandomCrop(self.image_res, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])

        # roidb_ni_fname = '/dccstor/jsdata1/data/inloc_roidb_ni_s.pkl'
        # with open(roidb_ni_fname, 'rb')as fid:
        #     self.roidb_ni = pickle.load(fid)

    def __getitem__(self, index):
        xml, label = self.data[index], self.labels[index]
        # if False:  # check class name
        #     train_classes = open(self.training_classes_list_path).read().splitlines()
        #     test_classes = open(self.test_classes_list_path).read().splitlines()
        #     val_classes = open("/dccstor/jsdata1/data/inloc_Amit_val_classes.txt").read().splitlines()
        #     n_code = os.path.basename(os.path.dirname(xml))
        #     class_name = folder_name_to_class_name[n_code]
        #     grp_cnt = 0
        #     if class_name in train_classes:
        #         print('getting image from train classes')
        #         grp_cnt+=1
        #     if class_name in test_classes:
        #         print('getting image from test classes')
        #         grp_cnt += 1
        #     if class_name in val_classes:
        #         print('getting image from val classes')
        #         grp_cnt += 1
        #     if grp_cnt==0:
        #         print('image came from unidentified class list !!!!!!!!!!!!!!!!!!!!!!')
        #     if grp_cnt>1:
        #         print('image came from more than one class list !!!!!!!!!!!!!!!!!!!!!!')

        if self.phase == 'test':
            img = xml.replace('.xml','.JPEG').replace('Annotations/CLS-LOC','Data/CLS-LOC')

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            gt_entry, img = load_imagenet_annotation(xml, img, label)
            # show_gt_boxes(img,gt_entry['boxes'],[str(i) for i in gt_entry['gt_classes']],save_file_path='/dccstor/jsdata1/dev/tmp.jpg')
            # DB_entry['gt_classes'],DB_entry['boxes'],DB_entry['gt_names']
        else:
            from PIL import Image
            # im = Image.open(img_filename)
            img = Image.open(xml).convert('RGB')

            gt_entry = None

        old_size = img.size #[columns, rows]
        if self.crop_style==1:
            larger = 0 if img.size[0] > img.size[1] else 1
            new_size = int((self.image_res * img.size[abs(larger - 1)]) / img.size[larger])
            resize_transform = transforms.Resize(new_size)
            img = resize_transform(img)
            if gt_entry is not None:
                gt_entry['boxes'] = gt_entry['boxes'].astype('float')
                gt_entry['boxes']*=float(new_size)/float(min(old_size))
                gt_entry['boxes'] = gt_entry['boxes'].round().astype('int16')
            pad_width_left = int((self.image_res - img.size[0]) / 2)
            pad_width_right = int(pad_width_left + ((self.image_res - img.size[0]) % 2))
            pad_width_top = int((self.image_res - img.size[1]) / 2)
            pad_width_bottom = int(pad_width_top + ((self.image_res - img.size[1]) % 2))
            pad_transform = transforms.Pad((pad_width_left, pad_width_top, pad_width_right, pad_width_bottom),
                                           padding_mode=self.pad_mode)
            img = pad_transform(img)
            if gt_entry is not None:
                gt_entry['boxes'][:, [0, 2]]+= pad_width_left
                gt_entry['boxes'][:, [1, 3]] += pad_width_top
        elif self.crop_style==2:
            downsize_tfm = transforms.Compose([
                transforms.Resize(self.image_res),  # resize to have the smaller dimension equal to image_res
                transforms.CenterCrop(self.image_res) # crop the other dimension to image_res
            ])
            img = downsize_tfm(img)
            img_scale = float(self.image_res)/float(min(old_size))
            if gt_entry is not None:
                gt_entry['boxes'] = gt_entry['boxes'].astype('float')*img_scale
            if old_size[1]>old_size[0]: #rows>columns, portrait. crop top-bottom. img_size is now [img_scale*old_size[1],image_res]
                crop_height= int((img_scale*old_size[1] - self.image_res) / 2)
                if gt_entry is not None:
                    gt_entry['boxes'][:, [1, 3]] -= crop_height
            else: # crop left-right
                crop_width = int((img_scale*old_size[0] -self.image_res) / 2)
                if gt_entry is not None:
                    gt_entry['boxes'][:, [0, 2]] -= crop_width
            if gt_entry is not None:
                gt_entry['boxes'] = gt_entry['boxes'].round().astype('int16')
        else:
            os.error('Imagenet_loc __getitem__: Unrecognized crop stype')

        #show_gt_boxes(img, gt_entry['boxes'], [str(i) for i in gt_entry['gt_classes']], save_file_path='/dccstor/jsdata1/dev/tmp.jpg')
        if self.transform is not None:
            img = self.transform(img)

        if gt_entry is not None:
            boxes = np.pad(gt_entry['boxes'],((0,10-gt_entry['boxes'].shape[0]),(0,0)),'constant')
        else:
            boxes = None

        if self.phase == 'test':
            return img, label, torch.tensor(boxes)
        else:
            return img, label

    def __len__(self):
        return len(self.data)

    def prepare_data_and_labels_train(self):
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize])

        # get train classes indices
        ClassIdxDictTrainTrain = load_data('./datasets/ImageNet/ImageNetClassIdxDictTrainCatTrainPhase.pkl',)
        # save train cat train phase data to file
        data_train_train = []
        labels_train_train = []
        for i in ClassIdxDictTrainTrain.keys():
            for j, idx in enumerate(ClassIdxDictTrainTrain[i]):
                path, label = self.samples[idx]
                data_train_train += [path]
                labels_train_train += [label]
                if self.debug and j == 50:
                    break
        print("data train cat train phase shape: {}".format(len(data_train_train)))
        print("labels train cat train phase shape: {}".format(len(labels_train_train)))
        self.data = data_train_train
        self.labels = labels_train_train

    def prepare_data_and_labels_val(self):
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize])

        # get train classes indices
        #ClassIdxDictTrainVal = load_data('/dccstor/alfassy/StarNet/data/ImageNetClassIdxDictTrainCatValPhase.pkl')
        ClassIdxDictTrainVal = load_data('./datasets/ImageNet/ImageNetClassIdxDictTrainCatValPhase.pkl')
        # save train cat train phase data to file
        data_train_val = []
        labels_train_val = []
        for i in ClassIdxDictTrainVal.keys():
            for j, idx in enumerate(ClassIdxDictTrainVal[i]):
                path, label = self.samples[idx]
                data_train_val += [path]
                labels_train_val += [label]
                if self.debug and j == 50:
                    break
        print("data train cat val phase shape: {}".format(len(data_train_val)))
        print("labels train cat val phase shape: {}".format(len(labels_train_val)))
        # get validation classes indices
        # val_indices = load_data('/dccstor/alfassy/StarNet/data/ImageNetValClasses.pkl')
        # ClassIdxDict = load_data('/dccstor/alfassy/StarNet/data/ImageNetClassIdxDict.pkl'

        # save val cat val phase data to file
        data_val = []
        labels_val = []
        for i in val_indices:
            for j, idx in enumerate(ClassIdxDict[i]):
                path, label = self.samples[idx]
                data_val += [path]
                labels_val += [label]
                if self.debug and j == 50:
                    break
        print("data val shape: {}".format(len(data_val)))
        print("labels val shape: {}".format(len(labels_val)))

        return data_train_val, labels_train_val, data_val, labels_val

    def prepare_data_and_labels_test(self):
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize])

        # get train classes indices
        base_folder = assert_folder('./datasets/ImageNet')
        train_classes_path = os.path.join(base_folder, 'ImageNetTrainClasses.pkl')
        ClassIdxDict_Test_path = os.path.join(base_folder, 'ImageNetClassIdxDictTest.pkl')
        TestClasses_path = os.path.join(base_folder, 'ImageNetTestClasses.pkl')
        train_indices = load_data(train_classes_path)
        ClassIdxDictTest = load_data(ClassIdxDict_Test_path)
        # save train cat train phase data to file
        data_test_train = []
        labels_test_train = []
        for i in train_indices:
            for j, idx in enumerate(ClassIdxDictTest[i]):
                path, label = self.samples[idx]
                data_test_train += [path]
                labels_test_train += [label]
                if self.debug and j == 50:
                    break
        print("data train cat test phase shape: {}".format(len(data_test_train)))
        print("labels train cat test phase shape: {}".format(len(labels_test_train)))
        # get validation classes indices
        test_indices = load_data(TestClasses_path)
        # save val cat val phase data to file
        data_test_test = []
        labels_test_test = []
        for i in test_indices:
            for j, idx in enumerate(ClassIdxDictTest[i]):
                path, label = self.samples[idx]
                data_test_test += [path]
                labels_test_test += [label]

        print("data val shape: {}".format(len(data_test_test)))
        print("labels val shape: {}".format(len(labels_test_test)))

        return data_test_train, labels_test_train, data_test_test, labels_test_test

    def create_data_dictionaries_train_val(self):
        base_folder = assert_folder('./datasets/ImageNet')
        #val_classes_list_path = os.path.join(base_folder,'imagenet_val_classes.txt')
        train_classes_path = os.path.join(base_folder,'ImageNetTrainClasses.pkl')
        ClassIdxDict_TrainCatTrainPhase_path = os.path.join(base_folder,'ImageNetClassIdxDictTrainCatTrainPhase.pkl')
        ClassIdxDict_TrainCatValPhase_path = os.path.join(base_folder,'ImageNetClassIdxDictTrainCatValPhase.pkl')
        IdxDict_path = os.path.join(base_folder,'ImageNetClassIdxDict.pkl')
        ValClasses_path = os.path.join(base_folder,'ImageNetValClasses.pkl')
        train_classes_joseph = open(self.training_classes_list_path).read().splitlines()
        # get train classes indices
        train_indices = []
        for class_name_joseph in train_classes_joseph:
            for folder_name, class_name in folder_name_to_class_name.items():
                if class_name_joseph == class_name:
                    train_indices += [self.class_to_idx[folder_name]]
        print("train indices num: {}".format(len(train_indices)))
        print("train classes list len: {}".format(len(train_classes_joseph)))
        with open(train_classes_path, 'wb') as f:
            pickle.dump(train_indices, f)
        f.close()
        print(train_indices)
        # Dictionary - Keys: labels(int), Values: list of indices of images tagged as $key
        ClassIdxDict = {el: [] for el in range(1000)}
        for idx, (path, label) in enumerate(self.samples):
            ClassIdxDict[label] += [idx]
        # divide train data between train and val phase
        random.seed(0)
        ClassIdxDictTrainTrain = {el: [] for el in train_indices}
        ClassIdxDictTrainVal = {el: [] for el in train_indices}
        for label in train_indices:
            ClassIdxDictTrainVal[label] = random.sample(ClassIdxDict[label], int(len(ClassIdxDict[label]) * 0.1))
            ClassIdxDictTrainTrain[label] = [idx for idx in ClassIdxDict[label] if
                                             idx not in ClassIdxDictTrainVal[label]]
        with open(ClassIdxDict_TrainCatTrainPhase_path, 'wb') as f:
            pickle.dump(ClassIdxDictTrainTrain, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDictTrainTrain.keys())
        with open(ClassIdxDict_TrainCatValPhase_path, 'wb') as f:
            pickle.dump(ClassIdxDictTrainVal, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDictTrainVal.keys())
        with open(IdxDict_path, 'wb') as f:
            pickle.dump(ClassIdxDict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDict.keys())
        # val_classes_list = open(val_classes_list_path).read().splitlines()
        # val_indices = []
        # for class_name_val in val_classes_list:
        #     for folder_name, class_name in folder_name_to_class_name.items():
        #         if class_name_val == class_name:
        #             val_indices += [self.class_to_idx[folder_name]]
        # print("val indices num: {}".format(len(val_indices)))
        # print("val classes num: {}".format(len(val_classes_list)))
        # with open(ValClasses_path, 'wb') as f:
        #     pickle.dump(val_indices, f)
        # f.close()
        # print(val_indices)

    def create_data_dictionaries_test(self):
        base_folder = assert_folder('./datasets/ImageNet')
        # load train classes indices
        train_classes_path = os.path.join(base_folder, 'ImageNetTrainClasses.pkl')
        ClassIdxDict_Test_path = os.path.join(base_folder, 'ImageNetClassIdxDictTest.pkl')
        TestClasses_path = os.path.join(base_folder,'ImageNetTestClasses.pkl')
        train_indices = load_data(train_classes_path)
        print(train_indices)
        # Dictionary - Keys: labels(int), Values: list of indices of images tagged as $key
        ClassIdxDictTest = {el: [] for el in range(1000)}
        for idx, (path, label) in enumerate(self.samples):
            ClassIdxDictTest[label] += [idx]
        with open(ClassIdxDict_Test_path, 'wb') as f:
            pickle.dump(ClassIdxDictTest, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print(ClassIdxDictTest.keys())
        test_classes_list = open(self.test_classes_list_path).read().splitlines()
        test_indices = []
        for class_name_test in test_classes_list:
            if class_name_test == "crane":
                test_indices += [134]
            else:
                for folder_name, class_name in folder_name_to_class_name.items():
                    if class_name_test == class_name:
                        test_indices += [self.class_to_idx[folder_name]]
                        break
        print("test indices num: {}".format(len(test_indices)))
        print("test classes num: {}".format(len(test_classes_list)))
        with open(TestClasses_path, 'wb') as f:
            pickle.dump(test_indices, f)
        f.close()
        print(test_indices)

    def create_ImageNet_Train_val_data(self):
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.image_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize])

        training_classes_list_path = \
            "/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_pascal_imagenet_15/inloc_first101_categories.txt"
        val_classes_list_path = '/dccstor/alfassy/StarNet/data/imagenet_val_classes.txt'

        train_classes_joseph = open(training_classes_list_path).read().splitlines()
        # get train classes indices
        train_indices = []
        for class_name_joseph in train_classes_joseph:
            for folder_name, class_name in folder_name_to_class_name.iteritems():
                if class_name_joseph == class_name:
                    train_indices += [self.class_to_idx[folder_name]]
        print("train indices num: {}".format(len(train_indices)))
        print("train classes list len: {}".format(len(train_classes_joseph)))
        with open('/dccstor/alfassy/StarNet/data/ImageNetTrainClasses.pkl', 'wb') as f:
            pickle.dump(train_indices, f)
        f.close()

        # Dictionary - Keys: labels(int), Values: list of indices of images tagged as $key
        ClassIdxDict = {el: [] for el in range(1000)}
        for idx, (path, label) in enumerate(self.samples):
            ClassIdxDict[label] += [idx]
        # divide train data between train and val phase
        random.seed(0)
        ClassIdxDictTrainTrain = {el: [] for el in range(1000)}
        ClassIdxDictTrainVal = {el: [] for el in range(1000)}
        for label in train_indices:
            ClassIdxDictTrainVal[label] = random.sample(ClassIdxDict[label], int(len(ClassIdxDict[label]) * 0.1))
            ClassIdxDictTrainTrain[label] = [idx for idx in ClassIdxDict[label] if idx not in ClassIdxDictTrainVal[label]]
        # save train cat train phase data to file
        transform = transforms.RandomResizedCrop(84)
        data_train_train = []
        labels_train_train = []
        for i in train_indices:
            for idx in ClassIdxDictTrainTrain[i]:
                path, label = self.samples[idx]
                image = PILI.open(path).convert('RGB')
                if transform is not None:
                    image = transform(image)
                image = np.array(image, dtype=np.uint8)
                data_train_train += [image]
                labels_train_train += [label]
        data_train_train = np.array(data_train_train, dtype=np.uint8)
        print("data train cat train phase shape: {}".format(data_train_train.shape))
        print("labels train cat train phase shape: {}".format(len(labels_train_train)))
        np.save('/dccstor/alfassy/StarNet/data/ImageNetTrainCatTrainPhaseData.pkl', data_train_train)
        with open('/dccstor/alfassy/StarNet/data/ImageNetTrainCatTrainPhaseLabels.pkl', 'wb') as f:
            pickle.dump(labels_train_train, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        # save train cat val phase data to file
        data_train_val = []
        labels_train_val = []
        for i in train_indices:
            for idx in ClassIdxDictTrainVal[i]:
                path, label = self.samples[idx]
                image = PILI.open(path).convert('RGB')
                if transform is not None:
                    image = transform(image)
                image = np.array(image, dtype=np.uint8)
                data_train_val += [image]
                labels_train_val += [label]
        data_train_val = np.array(data_train_val, dtype=np.uint8)
        print("data train cat val phase shape: {}".format(data_train_val.shape))
        print("labels train cat val phase shape: {}".format(len(labels_train_val)))
        np.save('/dccstor/alfassy/StarNet/data/ImageNetTrainCatValPhaseData.pkl', data_train_val)
        with open('/dccstor/alfassy/StarNet/data/ImageNetTrainCatValPhaseLabels.pkl', 'wb') as f:
            pickle.dump(labels_train_val, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        # get validation classes indices
        val_classes_list = open(val_classes_list_path).read().splitlines()
        val_indices = []
        for class_name_val in val_classes_list:
            for folder_name, class_name in folder_name_to_class_name.iteritems():
                if class_name_val == class_name:
                    val_indices += [self.class_to_idx[folder_name]]
        print("val indices num: {}".format(len(val_indices)))
        print("val classes num: {}".format(len(val_classes_list)))
        with open('/dccstor/alfassy/StarNet/data/ImageNetValClasses.pkl', 'wb') as f:
            pickle.dump(val_indices, f)
        f.close()
        # save val cat val phase data to file
        data_val = []
        labels_val = []
        for i in val_indices:
            for idx in ClassIdxDict[i]:
                path, label = self.samples[idx]
                image = PILI.open(path).convert('RGB')
                if transform is not None:
                    image = transform(image)
                data_val += [np.array(image, dtype=np.uint8)]
                labels_val += [label]
        data_val = np.array(data_val, dtype=np.uint8)
        print("data val shape: {}".format(data_val.shape))
        print("labels val shape: {}".format(len(labels_val)))
        np.save('/dccstor/alfassy/StarNet/data/ImageNetValData.pkl', data_val)
        with open('/dccstor/alfassy/StarNet/data/ImageNetValLabels.pkl', 'wb') as f:
            pickle.dump(labels_val, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


torch.manual_seed(2809)
torch.backends.cudnn.deterministic = True # Not necessary in this example

class FewShotDataloader():
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
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase == 'train'
                                else self.dataset.num_cats_novel)
        assert (nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase == 'train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert (nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
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
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set == 'base':
            labelIds = self.dataset.labelIds_base
        elif cat_set == 'novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert (len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert (nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel + nKbase)
            assert (len(cats_ids) == (nKnovel + nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert (len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.

        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert ((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase + Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase + Knovel_idx) for img_id in imds_ememplars]
        assert (len(Tnovel) == nTestNovel)
        assert (len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData_orig(self, examples):
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

    def createExamplesTensorData(self, examples):
        s_data = [self.dataset[img_idx] for img_idx, _ in examples]
        images = torch.stack([entry[0] for entry in s_data], dim=0)
        #labels = torch.LongTensor([entry[1] for entry in s_data])
        labels = torch.LongTensor([label for _, label in examples])
        if self.dataset.phase == 'test':
            gt_boxes = torch.stack([entry[2] for entry in s_data], dim=0)
            return images, labels, gt_boxes
        else:
            return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            np.random.seed(iter_idx)
            random.seed(iter_idx)
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            if self.dataset.phase == 'test':
                Xt, Yt, Bt = self.createExamplesTensorData(Test) # Images, Labels, GT_boxes
                Kall = torch.LongTensor(Kall)
                if len(Exemplars) > 0:
                    Xe, Ye, Be = self.createExamplesTensorData(Exemplars)
                    return Xe, Ye, Be, Xt, Yt, Bt, Kall, nKbase
                else:
                    return Xt, Yt, Bt, Kall, nKbase
            else:
                Xt, Yt = self.createExamplesTensorData(Test)  # Images, Labels, GT_boxes
                Kall = torch.LongTensor(Kall)
                if len(Exemplars) > 0:
                    Xe, Ye = self.createExamplesTensorData(Exemplars)
                    return Xe, Ye, Xt, Yt, Kall, nKbase
                else:
                    return Xt, Yt, Kall, nKbase

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

class FewShotDataloaderRepmet(FewShotDataloader):
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
        super(FewShotDataloaderRepmet, self).__init__(dataset,
                 nKnovel,  # number of novel categories.
                 nKbase,  # number of base categories.
                 nExemplars,  # number of training examples per novel category.
                 nTestNovel,  # number of test examples for all the novel categories.
                 nTestBase,  # number of test examples for all the base categories.
                 batch_size,  # number of training episodes per batch.
                 num_workers,
                 epoch_size,)

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            # Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt = self.createExamplesTensorData(Test)
            if len(Exemplars) > 0:
                Xe, Ye = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, Xt, Yt
            else:
                raise NotImplementedError("shouldnt get here")

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.dataset.__len__()), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

class ImageNetTestRepmet(ImageFolder):
    def __init__(self, options, do_not_use_random_transf=False, debug=False):
        super(ImageNetTestRepmet, self).__init__(os.path.join(imagenet_loc_data_path, 'train'))
        idx_to_class = {}
        for key in self.class_to_idx.keys():
            idx_to_class[self.class_to_idx[key]] = key
        self.idx_to_class = idx_to_class
        self.options = options
        self.test_n_support = options.test_way * options.test_shot
        episodes = load_data(joseph_test_episode_path)
        for epi in episodes:
            if len(epi['train_boxes']) < 5:
                episodes.remove(epi)
        self.episodes = episodes
        self.image_res = options.image_res
        assert((options.pad_mode == 'constant') or (options.pad_mode == 'edge') or (options.pad_mode == 'reflect') or
               (options.pad_mode == 'symmetric'))
        self.pad_mode = options.pad_mode
        self.debug = debug
        class_name_to_idx = {}
        for folder_name, class_name in folder_name_to_class_name.items():
            if class_name == "crane":
                class_name_to_idx[class_name.lower()] = 134
            else:
                class_name_to_idx[class_name.lower()] = self.class_to_idx[folder_name]
        self.class_name_to_idx = class_name_to_idx

        print('Loading ImageNet dataset - phase test')

        # mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        # std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if do_not_use_random_transf == True:
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.RandomCrop(self.image_res, padding=8),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        self.images_for_bbox_gen_transform = transforms.Compose([
            # transforms.RandomCrop(self.image_res, padding=8),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        # resize the image so that the bigger edge is self.image_res,
        # then pad the smaller edge to size self.image_res
        episode = self.episodes[index]
        if len(episode['train_boxes']) != self.test_n_support:
            episode = self.fix_episode(episode)
        # generate support batch
        # torchvision.utils.save_image(img, '/dccstor/alfassy/tmp/tmp.jpeg')
        # img.save("/dccstor/alfassy/tmp/tmp.jpeg", "jpeg")
        label_to_label_number = {self.class_name_to_idx[class_name.lower()]: i for i, class_name in enumerate(episode['epi_cats_names'])}
        labels_support = []
        for i, train_box in enumerate(episode['train_boxes']):
            img_path = train_box[2]
            sample_label = self.class_to_idx[(img_path.split('/')[-1]).split('_')[0]]
            labels_support += [label_to_label_number[sample_label]]
            img = PILI.open(str(imagenet_loc_data_path) + str(img_path.split('CLS-LOC')[1])).convert('RGB')
            if img.size[0] > img.size[1]:
                bigger = 0
            else:
                bigger = 1
            resize_transform = transforms.Resize(int((self.image_res * img.size[abs(bigger - 1)]) / img.size[bigger]))
            img = resize_transform(img)
            pad_width_left = int((self.image_res - img.size[0]) / 2)
            pad_width_right = int(pad_width_left + ((self.image_res - img.size[0]) % 2))
            pad_width_top = int((self.image_res - img.size[1]) / 2)
            pad_width_bottom = int(pad_width_top + ((self.image_res - img.size[1]) % 2))
            pad_transform = transforms.Pad((pad_width_left, pad_width_top, pad_width_right, pad_width_bottom),
                                           padding_mode=self.pad_mode)
            img = pad_transform(img)
            img_for_bbox_gen = self.images_for_bbox_gen_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            if i == 0:
                support_images = img_for_bbox_gen.unsqueeze(0)
                data_support = img.unsqueeze(0)
            else:
                data_support = torch.cat((data_support, img.unsqueeze(0)), dim=0)
                support_images = torch.cat((support_images, img_for_bbox_gen.unsqueeze(0)), dim=0)
        labels_support_torch = torch.LongTensor([label for label in labels_support])

        # generate query batch
        labels_query = []
        for i, query_image_path in enumerate(episode['query_images']):
            sample_label = self.class_to_idx[(query_image_path.split('/')[-1]).split('_')[0]]
            labels_query += [label_to_label_number[sample_label]]
            img = PILI.open(str(imagenet_loc_data_path) + str(query_image_path.split('CLS-LOC')[1])).convert('RGB')
            if img.size[0] > img.size[1]:
                bigger = 0
            else:
                bigger = 1
            resize_transform = transforms.Resize(
                int((self.image_res * img.size[abs(bigger - 1)]) / img.size[bigger]))
            img = resize_transform(img)
            pad_width_left = int((self.image_res - img.size[0]) / 2)
            pad_width_right = int(pad_width_left + ((self.image_res - img.size[0]) % 2))
            pad_width_top = int((self.image_res - img.size[1]) / 2)
            pad_width_bottom = int(pad_width_top + ((self.image_res - img.size[1]) % 2))
            pad_transform = transforms.Pad((pad_width_left, pad_width_top, pad_width_right, pad_width_bottom),
                                           padding_mode=self.pad_mode)
            img = pad_transform(img)
            img_for_bbox_gen = self.images_for_bbox_gen_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            if i == 0:
                query_images = img_for_bbox_gen.unsqueeze(0)
                data_query = img.unsqueeze(0)
            else:
                data_query = torch.cat((data_query, img.unsqueeze(0)), dim=0)
                query_images = torch.cat((query_images, img_for_bbox_gen.unsqueeze(0)), dim=0)

        labels_query_torch = torch.LongTensor([label for label in labels_query])

        return data_support, labels_support_torch, data_query, labels_query_torch, support_images, query_images

    def __len__(self):
        return len(self.episodes)

    def fix_episode(self, episode2fix):
        '''
        Takes episodes with redundant support examples and removes them.
        :return: episode
        '''

        train_boxes = []
        seen_categories = []
        for element in episode2fix['train_boxes']:
            cat = element[0]
            if cat not in seen_categories:
                seen_categories += [cat]
                train_boxes += [element]
        episode2fix['train_boxes'] = train_boxes
        return episode2fix

folder_name_to_class_name = {"n01440764": "tench, Tinca tinca",
"n01443537": "goldfish, Carassius auratus",
"n01484850": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
"n01491361": "tiger shark, Galeocerdo cuvieri",
"n01494475": "hammerhead, hammerhead shark",
"n01496331": "electric ray, crampfish, numbfish, torpedo",
"n01498041": "stingray",
"n01514668": "cock",
"n01514859": "hen",
"n01518878": "ostrich, Struthio camelus",
"n01530575": "brambling, Fringilla montifringilla",
"n01531178": "goldfinch, Carduelis carduelis",
"n01532829": "house finch, linnet, Carpodacus mexicanus",
"n01534433": "junco, snowbird",
"n01537544": "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
"n01558993": "robin, American robin, Turdus migratorius",
"n01560419": "bulbul",
"n01580077": "jay",
"n01582220": "magpie",
"n01592084": "chickadee",
"n01601694": "water ouzel, dipper",
"n01608432": "kite",
"n01614925": "bald eagle, American eagle, Haliaeetus leucocephalus",
"n01616318": "vulture",
"n01622779": "great grey owl, great gray owl, Strix nebulosa",
"n01629819": "European fire salamander, Salamandra salamandra",
"n01630670": "common newt, Triturus vulgaris",
"n01631663": "eft",
"n01632458": "spotted salamander, Ambystoma maculatum",
"n01632777": "axolotl, mud puppy, Ambystoma mexicanum",
"n01641577": "bullfrog, Rana catesbeiana",
"n01644373": "tree frog, tree-frog",
"n01644900": "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
"n01664065": "loggerhead, loggerhead turtle, Caretta caretta",
"n01665541": "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
"n01667114": "mud turtle",
"n01667778": "terrapin",
"n01669191": "box turtle, box tortoise",
"n01675722": "banded gecko",
"n01677366": "common iguana, iguana, Iguana iguana",
"n01682714": "American chameleon, anole, Anolis carolinensis",
"n01685808": "whiptail, whiptail lizard",
"n01687978": "agama",
"n01688243": "frilled lizard, Chlamydosaurus kingi",
"n01689811": "alligator lizard",
"n01692333": "Gila monster, Heloderma suspectum",
"n01693334": "green lizard, Lacerta viridis",
"n01694178": "African chameleon, Chamaeleo chamaeleon",
"n01695060": "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
"n01697457": "African crocodile, Nile crocodile, Crocodylus niloticus",
"n01698640": "American alligator, Alligator mississipiensis",
"n01704323": "triceratops",
"n01728572": "thunder snake, worm snake, Carphophis amoenus",
"n01728920": "ringneck snake, ring-necked snake, ring snake",
"n01729322": "hognose snake, puff adder, sand viper",
"n01729977": "green snake, grass snake",
"n01734418": "king snake, kingsnake",
"n01735189": "garter snake, grass snake",
"n01737021": "water snake",
"n01739381": "vine snake",
"n01740131": "night snake, Hypsiglena torquata",
"n01742172": "boa constrictor, Constrictor constrictor",
"n01744401": "rock python, rock snake, Python sebae",
"n01748264": "Indian cobra, Naja naja",
"n01749939": "green mamba",
"n01751748": "sea snake",
"n01753488": "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
"n01755581": "diamondback, diamondback rattlesnake, Crotalus adamanteus",
"n01756291": "sidewinder, horned rattlesnake, Crotalus cerastes",
"n01768244": "trilobite",
"n01770081": "harvestman, daddy longlegs, Phalangium opilio",
"n01770393": "scorpion",
"n01773157": "black and gold garden spider, Argiope aurantia",
"n01773549": "barn spider, Araneus cavaticus",
"n01773797": "garden spider, Aranea diademata",
"n01774384": "black widow, Latrodectus mactans",
"n01774750": "tarantula",
"n01775062": "wolf spider, hunting spider",
"n01776313": "tick",
"n01784675": "centipede",
"n01795545": "black grouse",
"n01796340": "ptarmigan",
"n01797886": "ruffed grouse, partridge, Bonasa umbellus",
"n01798484": "prairie chicken, prairie grouse, prairie fowl",
"n01806143": "peacock",
"n01806567": "quail",
"n01807496": "partridge",
"n01817953": "African grey, African gray, Psittacus erithacus",
"n01818515": "macaw",
"n01819313": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
"n01820546": "lorikeet",
"n01824575": "coucal",
"n01828970": "bee eater",
"n01829413": "hornbill",
"n01833805": "hummingbird",
"n01843065": "jacamar",
"n01843383": "toucan",
"n01847000": "drake",
"n01855032": "red-breasted merganser, Mergus serrator",
"n01855672": "goose",
"n01860187": "black swan, Cygnus atratus",
"n01871265": "tusker",
"n01872401": "echidna, spiny anteater, anteater",
"n01873310": "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
"n01877812": "wallaby, brush kangaroo",
"n01882714": "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
"n01883070": "wombat",
"n01910747": "jellyfish",
"n01914609": "sea anemone, anemone",
"n01917289": "brain coral",
"n01924916": "flatworm, platyhelminth",
"n01930112": "nematode, nematode worm, roundworm",
"n01943899": "conch",
"n01944390": "snail",
"n01945685": "slug",
"n01950731": "sea slug, nudibranch",
"n01955084": "chiton, coat-of-mail shell, sea cradle, polyplacophore",
"n01968897": "chambered nautilus, pearly nautilus, nautilus",
"n01978287": "Dungeness crab, Cancer magister",
"n01978455": "rock crab, Cancer irroratus",
"n01980166": "fiddler crab",
"n01981276": "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
"n01983481": "American lobster, Northern lobster, Maine lobster, Homarus americanus",
"n01984695": "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
"n01985128": "crayfish, crawfish, crawdad, crawdaddy",
"n01986214": "hermit crab",
"n01990800": "isopod",
"n02002556": "white stork, Ciconia ciconia",
"n02002724": "black stork, Ciconia nigra",
"n02006656": "spoonbill",
"n02007558": "flamingo",
"n02009229": "little blue heron, Egretta caerulea",
"n02009912": "American egret, great white heron, Egretta albus",
"n02011460": "bittern",
"n02012849": "crane",
"n02013706": "limpkin, Aramus pictus",
"n02017213": "European gallinule, Porphyrio porphyrio",
"n02018207": "American coot, marsh hen, mud hen, water hen, Fulica americana",
"n02018795": "bustard",
"n02025239": "ruddy turnstone, Arenaria interpres",
"n02027492": "red-backed sandpiper, dunlin, Erolia alpina",
"n02028035": "redshank, Tringa totanus",
"n02033041": "dowitcher",
"n02037110": "oystercatcher, oyster catcher",
"n02051845": "pelican",
"n02056570": "king penguin, Aptenodytes patagonica",
"n02058221": "albatross, mollymawk",
"n02066245": "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
"n02071294": "killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
"n02074367": "dugong, Dugong dugon",
"n02077923": "sea lion",
"n02085620": "Chihuahua",
"n02085782": "Japanese spaniel",
"n02085936": "Maltese dog, Maltese terrier, Maltese",
"n02086079": "Pekinese, Pekingese, Peke",
"n02086240": "Shih-Tzu",
"n02086646": "Blenheim spaniel",
"n02086910": "papillon",
"n02087046": "toy terrier",
"n02087394": "Rhodesian ridgeback",
"n02088094": "Afghan hound, Afghan",
"n02088238": "basset, basset hound",
"n02088364": "beagle",
"n02088466": "bloodhound, sleuthhound",
"n02088632": "bluetick",
"n02089078": "black-and-tan coonhound",
"n02089867": "Walker hound, Walker foxhound",
"n02089973": "English foxhound",
"n02090379": "redbone",
"n02090622": "borzoi, Russian wolfhound",
"n02090721": "Irish wolfhound",
"n02091032": "Italian greyhound",
"n02091134": "whippet",
"n02091244": "Ibizan hound, Ibizan Podenco",
"n02091467": "Norwegian elkhound, elkhound",
"n02091635": "otterhound, otter hound",
"n02091831": "Saluki, gazelle hound",
"n02092002": "Scottish deerhound, deerhound",
"n02092339": "Weimaraner",
"n02093256": "Staffordshire bullterrier, Staffordshire bull terrier",
"n02093428": "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
"n02093647": "Bedlington terrier",
"n02093754": "Border terrier",
"n02093859": "Kerry blue terrier",
"n02093991": "Irish terrier",
"n02094114": "Norfolk terrier",
"n02094258": "Norwich terrier",
"n02094433": "Yorkshire terrier",
"n02095314": "wire-haired fox terrier",
"n02095570": "Lakeland terrier",
"n02095889": "Sealyham terrier, Sealyham",
"n02096051": "Airedale, Airedale terrier",
"n02096177": "cairn, cairn terrier",
"n02096294": "Australian terrier",
"n02096437": "Dandie Dinmont, Dandie Dinmont terrier",
"n02096585": "Boston bull, Boston terrier",
"n02097047": "miniature schnauzer",
"n02097130": "giant schnauzer",
"n02097209": "standard schnauzer",
"n02097298": "Scotch terrier, Scottish terrier, Scottie",
"n02097474": "Tibetan terrier, chrysanthemum dog",
"n02097658": "silky terrier, Sydney silky",
"n02098105": "soft-coated wheaten terrier",
"n02098286": "West Highland white terrier",
"n02098413": "Lhasa, Lhasa apso",
"n02099267": "flat-coated retriever",
"n02099429": "curly-coated retriever",
"n02099601": "golden retriever",
"n02099712": "Labrador retriever",
"n02099849": "Chesapeake Bay retriever",
"n02100236": "German short-haired pointer",
"n02100583": "vizsla, Hungarian pointer",
"n02100735": "English setter",
"n02100877": "Irish setter, red setter",
"n02101006": "Gordon setter",
"n02101388": "Brittany spaniel",
"n02101556": "clumber, clumber spaniel",
"n02102040": "English springer, English springer spaniel",
"n02102177": "Welsh springer spaniel",
"n02102318": "cocker spaniel, English cocker spaniel, cocker",
"n02102480": "Sussex spaniel",
"n02102973": "Irish water spaniel",
"n02104029": "kuvasz",
"n02104365": "schipperke",
"n02105056": "groenendael",
"n02105162": "malinois",
"n02105251": "briard",
"n02105412": "kelpie",
"n02105505": "komondor",
"n02105641": "Old English sheepdog, bobtail",
"n02105855": "Shetland sheepdog, Shetland sheep dog, Shetland",
"n02106030": "collie",
"n02106166": "Border collie",
"n02106382": "Bouvier des Flandres, Bouviers des Flandres",
"n02106550": "Rottweiler",
"n02106662": "German shepherd, German shepherd dog, German police dog, alsatian",
"n02107142": "Doberman, Doberman pinscher",
"n02107312": "miniature pinscher",
"n02107574": "Greater Swiss Mountain dog",
"n02107683": "Bernese mountain dog",
"n02107908": "Appenzeller",
"n02108000": "EntleBucher",
"n02108089": "boxer",
"n02108422": "bull mastiff",
"n02108551": "Tibetan mastiff",
"n02108915": "French bulldog",
"n02109047": "Great Dane",
"n02109525": "Saint Bernard, St Bernard",
"n02109961": "Eskimo dog, husky",
"n02110063": "malamute, malemute, Alaskan malamute",
"n02110185": "Siberian husky",
"n02110341": "dalmatian, coach dog, carriage dog",
"n02110627": "affenpinscher, monkey pinscher, monkey dog",
"n02110806": "basenji",
"n02110958": "pug, pug-dog",
"n02111129": "Leonberg",
"n02111277": "Newfoundland, Newfoundland dog",
"n02111500": "Great Pyrenees",
"n02111889": "Samoyed, Samoyede",
"n02112018": "Pomeranian",
"n02112137": "chow, chow chow",
"n02112350": "keeshond",
"n02112706": "Brabancon griffon",
"n02113023": "Pembroke, Pembroke Welsh corgi",
"n02113186": "Cardigan, Cardigan Welsh corgi",
"n02113624": "toy poodle",
"n02113712": "miniature poodle",
"n02113799": "standard poodle",
"n02113978": "Mexican hairless",
"n02114367": "timber wolf, grey wolf, gray wolf, Canis lupus",
"n02114548": "white wolf, Arctic wolf, Canis lupus tundrarum",
"n02114712": "red wolf, maned wolf, Canis rufus, Canis niger",
"n02114855": "coyote, prairie wolf, brush wolf, Canis latrans",
"n02115641": "dingo, warrigal, warragal, Canis dingo",
"n02115913": "dhole, Cuon alpinus",
"n02116738": "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
"n02117135": "hyena, hyaena",
"n02119022": "red fox, Vulpes vulpes",
"n02119789": "kit fox, Vulpes macrotis",
"n02120079": "Arctic fox, white fox, Alopex lagopus",
"n02120505": "grey fox, gray fox, Urocyon cinereoargenteus",
"n02123045": "tabby, tabby cat",
"n02123159": "tiger cat",
"n02123394": "Persian cat",
"n02123597": "Siamese cat, Siamese",
"n02124075": "Egyptian cat",
"n02125311": "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
"n02127052": "lynx, catamount",
"n02128385": "leopard, Panthera pardus",
"n02128757": "snow leopard, ounce, Panthera uncia",
"n02128925": "jaguar, panther, Panthera onca, Felis onca",
"n02129165": "lion, king of beasts, Panthera leo",
"n02129604": "tiger, Panthera tigris",
"n02130308": "cheetah, chetah, Acinonyx jubatus",
"n02132136": "brown bear, bruin, Ursus arctos",
"n02133161": "American black bear, black bear, Ursus americanus, Euarctos americanus",
"n02134084": "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
"n02134418": "sloth bear, Melursus ursinus, Ursus ursinus",
"n02137549": "mongoose",
"n02138441": "meerkat, mierkat",
"n02165105": "tiger beetle",
"n02165456": "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
"n02167151": "ground beetle, carabid beetle",
"n02168699": "long-horned beetle, longicorn, longicorn beetle",
"n02169497": "leaf beetle, chrysomelid",
"n02172182": "dung beetle",
"n02174001": "rhinoceros beetle",
"n02177972": "weevil",
"n02190166": "fly",
"n02206856": "bee",
"n02219486": "ant, emmet, pismire",
"n02226429": "grasshopper, hopper",
"n02229544": "cricket",
"n02231487": "walking stick, walkingstick, stick insect",
"n02233338": "cockroach, roach",
"n02236044": "mantis, mantid",
"n02256656": "cicada, cicala",
"n02259212": "leafhopper",
"n02264363": "lacewing, lacewing fly",
"n02268443": "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
"n02268853": "damselfly",
"n02276258": "admiral",
"n02277742": "ringlet, ringlet butterfly",
"n02279972": "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
"n02280649": "cabbage butterfly",
"n02281406": "sulphur butterfly, sulfur butterfly",
"n02281787": "lycaenid, lycaenid butterfly",
"n02317335": "starfish, sea star",
"n02319095": "sea urchin",
"n02321529": "sea cucumber, holothurian",
"n02325366": "wood rabbit, cottontail, cottontail rabbit",
"n02326432": "hare",
"n02328150": "Angora, Angora rabbit",
"n02342885": "hamster",
"n02346627": "porcupine, hedgehog",
"n02356798": "fox squirrel, eastern fox squirrel, Sciurus niger",
"n02361337": "marmot",
"n02363005": "beaver",
"n02364673": "guinea pig, Cavia cobaya",
"n02389026": "sorrel",
"n02391049": "zebra",
"n02395406": "hog, pig, grunter, squealer, Sus scrofa",
"n02396427": "wild boar, boar, Sus scrofa",
"n02397096": "warthog",
"n02398521": "hippopotamus, hippo, river horse, Hippopotamus amphibius",
"n02403003": "ox",
"n02408429": "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
"n02410509": "bison",
"n02412080": "ram, tup",
"n02415577": "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
"n02417914": "ibex, Capra ibex",
"n02422106": "hartebeest",
"n02422699": "impala, Aepyceros melampus",
"n02423022": "gazelle",
"n02437312": "Arabian camel, dromedary, Camelus dromedarius",
"n02437616": "llama",
"n02441942": "weasel",
"n02442845": "mink",
"n02443114": "polecat, fitch, foulmart, foumart, Mustela putorius",
"n02443484": "black-footed ferret, ferret, Mustela nigripes",
"n02444819": "otter",
"n02445715": "skunk, polecat, wood pussy",
"n02447366": "badger",
"n02454379": "armadillo",
"n02457408": "three-toed sloth, ai, Bradypus tridactylus",
"n02480495": "orangutan, orang, orangutang, Pongo pygmaeus",
"n02480855": "gorilla, Gorilla gorilla",
"n02481823": "chimpanzee, chimp, Pan troglodytes",
"n02483362": "gibbon, Hylobates lar",
"n02483708": "siamang, Hylobates syndactylus, Symphalangus syndactylus",
"n02484975": "guenon, guenon monkey",
"n02486261": "patas, hussar monkey, Erythrocebus patas",
"n02486410": "baboon",
"n02487347": "macaque",
"n02488291": "langur",
"n02488702": "colobus, colobus monkey",
"n02489166": "proboscis monkey, Nasalis larvatus",
"n02490219": "marmoset",
"n02492035": "capuchin, ringtail, Cebus capucinus",
"n02492660": "howler monkey, howler",
"n02493509": "titi, titi monkey",
"n02493793": "spider monkey, Ateles geoffroyi",
"n02494079": "squirrel monkey, Saimiri sciureus",
"n02497673": "Madagascar cat, ring-tailed lemur, Lemur catta",
"n02500267": "indri, indris, Indri indri, Indri brevicaudatus",
"n02504013": "Indian elephant, Elephas maximus",
"n02504458": "African elephant, Loxodonta africana",
"n02509815": "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
"n02510455": "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
"n02514041": "barracouta, snoek",
"n02526121": "eel",
"n02536864": "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
"n02606052": "rock beauty, Holocanthus tricolor",
"n02607072": "anemone fish",
"n02640242": "sturgeon",
"n02641379": "gar, garfish, garpike, billfish, Lepisosteus osseus",
"n02643566": "lionfish",
"n02655020": "puffer, pufferfish, blowfish, globefish",
"n02666196": "abacus",
"n02667093": "abaya",
"n02669723": "academic gown, academic robe, judge's robe",
"n02672831": "accordion, piano accordion, squeeze box",
"n02676566": "acoustic guitar",
"n02687172": "aircraft carrier, carrier, flattop, attack aircraft carrier",
"n02690373": "airliner",
"n02692877": "airship, dirigible",
"n02699494": "altar",
"n02701002": "ambulance",
"n02704792": "amphibian, amphibious vehicle",
"n02708093": "analog clock",
"n02727426": "apiary, bee house",
"n02730930": "apron",
"n02747177": "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
"n02749479": "assault rifle, assault gun",
"n02769748": "backpack, back pack, knapsack, packsack, rucksack, haversack",
"n02776631": "bakery, bakeshop, bakehouse",
"n02777292": "balance beam, beam",
"n02782093": "balloon",
"n02783161": "ballpoint, ballpoint pen, ballpen, Biro",
"n02786058": "Band Aid",
"n02787622": "banjo",
"n02788148": "bannister, banister, balustrade, balusters, handrail",
"n02790996": "barbell",
"n02791124": "barber chair",
"n02791270": "barbershop",
"n02793495": "barn",
"n02794156": "barometer",
"n02795169": "barrel, cask",
"n02797295": "barrow, garden cart, lawn cart, wheelbarrow",
"n02799071": "baseball",
"n02802426": "basketball",
"n02804414": "bassinet",
"n02804610": "bassoon",
"n02807133": "bathing cap, swimming cap",
"n02808304": "bath towel",
"n02808440": "bathtub, bathing tub, bath, tub",
"n02814533": "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
"n02814860": "beacon, lighthouse, beacon light, pharos",
"n02815834": "beaker",
"n02817516": "bearskin, busby, shako",
"n02823428": "beer bottle",
"n02823750": "beer glass",
"n02825657": "bell cote, bell cot",
"n02834397": "bib",
"n02835271": "bicycle-built-for-two, tandem bicycle, tandem",
"n02837789": "bikini, two-piece",
"n02840245": "binder, ring-binder",
"n02841315": "binoculars, field glasses, opera glasses",
"n02843684": "birdhouse",
"n02859443": "boathouse",
"n02860847": "bobsled, bobsleigh, bob",
"n02865351": "bolo tie, bolo, bola tie, bola",
"n02869837": "bonnet, poke bonnet",
"n02870880": "bookcase",
"n02871525": "bookshop, bookstore, bookstall",
"n02877765": "bottlecap",
"n02879718": "bow",
"n02883205": "bow tie, bow-tie, bowtie",
"n02892201": "brass, memorial tablet, plaque",
"n02892767": "brassiere, bra, bandeau",
"n02894605": "breakwater, groin, groyne, mole, bulwark, seawall, jetty",
"n02895154": "breastplate, aegis, egis",
"n02906734": "broom",
"n02909870": "bucket, pail",
"n02910353": "buckle",
"n02916936": "bulletproof vest",
"n02917067": "bullet train, bullet",
"n02927161": "butcher shop, meat market",
"n02930766": "cab, hack, taxi, taxicab",
"n02939185": "caldron, cauldron",
"n02948072": "candle, taper, wax light",
"n02950826": "cannon",
"n02951358": "canoe",
"n02951585": "can opener, tin opener",
"n02963159": "cardigan",
"n02965783": "car mirror",
"n02966193": "carousel, carrousel, merry-go-round, roundabout, whirligig",
"n02966687": "carpenter's kit, tool kit",
"n02971356": "carton",
"n02974003": "car wheel",
"n02977058": "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
"n02978881": "cassette",
"n02979186": "cassette player",
"n02980441": "castle",
"n02981792": "catamaran",
"n02988304": "CD player",
"n02992211": "cello, violoncello",
"n02992529": "cellular telephone, cellular phone, cellphone, cell, mobile phone",
"n02999410": "chain",
"n03000134": "chainlink fence",
"n03000247": "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
"n03000684": "chain saw, chainsaw",
"n03014705": "chest",
"n03016953": "chiffonier, commode",
"n03017168": "chime, bell, gong",
"n03018349": "china cabinet, china closet",
"n03026506": "Christmas stocking",
"n03028079": "church, church building",
"n03032252": "cinema, movie theater, movie theatre, movie house, picture palace",
"n03041632": "cleaver, meat cleaver, chopper",
"n03042490": "cliff dwelling",
"n03045698": "cloak",
"n03047690": "clog, geta, patten, sabot",
"n03062245": "cocktail shaker",
"n03063599": "coffee mug",
"n03063689": "coffeepot",
"n03065424": "coil, spiral, volute, whorl, helix",
"n03075370": "combination lock",
"n03085013": "computer keyboard, keypad",
"n03089624": "confectionery, confectionary, candy store",
"n03095699": "container ship, containership, container vessel",
"n03100240": "convertible",
"n03109150": "corkscrew, bottle screw",
"n03110669": "cornet, horn, trumpet, trump",
"n03124043": "cowboy boot",
"n03124170": "cowboy hat, ten-gallon hat",
"n03125729": "cradle",
"n03126707": "crane",
"n03127747": "crash helmet",
"n03127925": "crate",
"n03131574": "crib, cot",
"n03133878": "Crock Pot",
"n03134739": "croquet ball",
"n03141823": "crutch",
"n03146219": "cuirass",
"n03160309": "dam, dike, dyke",
"n03179701": "desk",
"n03180011": "desktop computer",
"n03187595": "dial telephone, dial phone",
"n03188531": "diaper, nappy, napkin",
"n03196217": "digital clock",
"n03197337": "digital watch",
"n03201208": "dining table, board",
"n03207743": "dishrag, dishcloth",
"n03207941": "dishwasher, dish washer, dishwashing machine",
"n03208938": "disk brake, disc brake",
"n03216828": "dock, dockage, docking facility",
"n03218198": "dogsled, dog sled, dog sleigh",
"n03220513": "dome",
"n03223299": "doormat, welcome mat",
"n03240683": "drilling platform, offshore rig",
"n03249569": "drum, membranophone, tympan",
"n03250847": "drumstick",
"n03255030": "dumbbell",
"n03259280": "Dutch oven",
"n03271574": "electric fan, blower",
"n03272010": "electric guitar",
"n03272562": "electric locomotive",
"n03290653": "entertainment center",
"n03291819": "envelope",
"n03297495": "espresso maker",
"n03314780": "face powder",
"n03325584": "feather boa, boa",
"n03337140": "file, file cabinet, filing cabinet",
"n03344393": "fireboat",
"n03345487": "fire engine, fire truck",
"n03347037": "fire screen, fireguard",
"n03355925": "flagpole, flagstaff",
"n03372029": "flute, transverse flute",
"n03376595": "folding chair",
"n03379051": "football helmet",
"n03384352": "forklift",
"n03388043": "fountain",
"n03388183": "fountain pen",
"n03388549": "four-poster",
"n03393912": "freight car",
"n03394916": "French horn, horn",
"n03400231": "frying pan, frypan, skillet",
"n03404251": "fur coat",
"n03417042": "garbage truck, dustcart",
"n03424325": "gasmask, respirator, gas helmet",
"n03425413": "gas pump, gasoline pump, petrol pump, island dispenser",
"n03443371": "goblet",
"n03444034": "go-kart",
"n03445777": "golf ball",
"n03445924": "golfcart, golf cart",
"n03447447": "gondola",
"n03447721": "gong, tam-tam",
"n03450230": "gown",
"n03452741": "grand piano, grand",
"n03457902": "greenhouse, nursery, glasshouse",
"n03459775": "grille, radiator grille",
"n03461385": "grocery store, grocery, food market, market",
"n03467068": "guillotine",
"n03476684": "hair slide",
"n03476991": "hair spray",
"n03478589": "half track",
"n03481172": "hammer",
"n03482405": "hamper",
"n03483316": "hand blower, blow dryer, blow drier, hair dryer, hair drier",
"n03485407": "hand-held computer, hand-held microcomputer",
"n03485794": "handkerchief, hankie, hanky, hankey",
"n03492542": "hard disc, hard disk, fixed disk",
"n03494278": "harmonica, mouth organ, harp, mouth harp",
"n03495258": "harp",
"n03496892": "harvester, reaper",
"n03498962": "hatchet",
"n03527444": "holster",
"n03529860": "home theater, home theatre",
"n03530642": "honeycomb",
"n03532672": "hook, claw",
"n03534580": "hoopskirt, crinoline",
"n03535780": "horizontal bar, high bar",
"n03538406": "horse cart, horse-cart",
"n03544143": "hourglass",
"n03584254": "iPod",
"n03584829": "iron, smoothing iron",
"n03590841": "jack-o'-lantern",
"n03594734": "jean, blue jean, denim",
"n03594945": "jeep, landrover",
"n03595614": "jersey, T-shirt, tee shirt",
"n03598930": "jigsaw puzzle",
"n03599486": "jinrikisha, ricksha, rickshaw",
"n03602883": "joystick",
"n03617480": "kimono",
"n03623198": "knee pad",
"n03627232": "knot",
"n03630383": "lab coat, laboratory coat",
"n03633091": "ladle",
"n03637318": "lampshade, lamp shade",
"n03642806": "laptop, laptop computer",
"n03649909": "lawn mower, mower",
"n03657121": "lens cap, lens cover",
"n03658185": "letter opener, paper knife, paperknife",
"n03661043": "library",
"n03662601": "lifeboat",
"n03666591": "lighter, light, igniter, ignitor",
"n03670208": "limousine, limo",
"n03673027": "liner, ocean liner",
"n03676483": "lipstick, lip rouge",
"n03680355": "Loafer",
"n03690938": "lotion",
"n03691459": "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
"n03692522": "loupe, jeweler's loupe",
"n03697007": "lumbermill, sawmill",
"n03706229": "magnetic compass",
"n03709823": "mailbag, postbag",
"n03710193": "mailbox, letter box",
"n03710637": "maillot",
"n03710721": "maillot, tank suit",
"n03717622": "manhole cover",
"n03720891": "maraca",
"n03721384": "marimba, xylophone",
"n03724870": "mask",
"n03729826": "matchstick",
"n03733131": "maypole",
"n03733281": "maze, labyrinth",
"n03733805": "measuring cup",
"n03742115": "medicine chest, medicine cabinet",
"n03743016": "megalith, megalithic structure",
"n03759954": "microphone, mike",
"n03761084": "microwave, microwave oven",
"n03763968": "military uniform",
"n03764736": "milk can",
"n03769881": "minibus",
"n03770439": "miniskirt, mini",
"n03770679": "minivan",
"n03773504": "missile",
"n03775071": "mitten",
"n03775546": "mixing bowl",
"n03776460": "mobile home, manufactured home",
"n03777568": "Model T",
"n03777754": "modem",
"n03781244": "monastery",
"n03782006": "monitor",
"n03785016": "moped",
"n03786901": "mortar",
"n03787032": "mortarboard",
"n03788195": "mosque",
"n03788365": "mosquito net",
"n03791053": "motor scooter, scooter",
"n03792782": "mountain bike, all-terrain bike, off-roader",
"n03792972": "mountain tent",
"n03793489": "mouse, computer mouse",
"n03794056": "mousetrap",
"n03796401": "moving van",
"n03803284": "muzzle",
"n03804744": "nail",
"n03814639": "neck brace",
"n03814906": "necklace",
"n03825788": "nipple",
"n03832673": "notebook, notebook computer",
"n03837869": "obelisk",
"n03838899": "oboe, hautboy, hautbois",
"n03840681": "ocarina, sweet potato",
"n03841143": "odometer, hodometer, mileometer, milometer",
"n03843555": "oil filter",
"n03854065": "organ, pipe organ",
"n03857828": "oscilloscope, scope, cathode-ray oscilloscope, CRO",
"n03866082": "overskirt",
"n03868242": "oxcart",
"n03868863": "oxygen mask",
"n03871628": "packet",
"n03873416": "paddle, boat paddle",
"n03874293": "paddlewheel, paddle wheel",
"n03874599": "padlock",
"n03876231": "paintbrush",
"n03877472": "pajama, pyjama, pj's, jammies",
"n03877845": "palace",
"n03884397": "panpipe, pandean pipe, syrinx",
"n03887697": "paper towel",
"n03888257": "parachute, chute",
"n03888605": "parallel bars, bars",
"n03891251": "park bench",
"n03891332": "parking meter",
"n03895866": "passenger car, coach, carriage",
"n03899768": "patio, terrace",
"n03902125": "pay-phone, pay-station",
"n03903868": "pedestal, plinth, footstall",
"n03908618": "pencil box, pencil case",
"n03908714": "pencil sharpener",
"n03916031": "perfume, essence",
"n03920288": "Petri dish",
"n03924679": "photocopier",
"n03929660": "pick, plectrum, plectron",
"n03929855": "pickelhaube",
"n03930313": "picket fence, paling",
"n03930630": "pickup, pickup truck",
"n03933933": "pier",
"n03935335": "piggy bank, penny bank",
"n03937543": "pill bottle",
"n03938244": "pillow",
"n03942813": "ping-pong ball",
"n03944341": "pinwheel",
"n03947888": "pirate, pirate ship",
"n03950228": "pitcher, ewer",
"n03954731": "plane, carpenter's plane, woodworking plane",
"n03956157": "planetarium",
"n03958227": "plastic bag",
"n03961711": "plate rack",
"n03967562": "plow, plough",
"n03970156": "plunger, plumber's helper",
"n03976467": "Polaroid camera, Polaroid Land camera",
"n03976657": "pole",
"n03977966": "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
"n03980874": "poncho",
"n03982430": "pool table, billiard table, snooker table",
"n03983396": "pop bottle, soda bottle",
"n03991062": "pot, flowerpot",
"n03992509": "potter's wheel",
"n03995372": "power drill",
"n03998194": "prayer rug, prayer mat",
"n04004767": "printer",
"n04005630": "prison, prison house",
"n04008634": "projectile, missile",
"n04009552": "projector",
"n04019541": "puck, hockey puck",
"n04023962": "punching bag, punch bag, punching ball, punchball",
"n04026417": "purse",
"n04033901": "quill, quill pen",
"n04033995": "quilt, comforter, comfort, puff",
"n04037443": "racer, race car, racing car",
"n04039381": "racket, racquet",
"n04040759": "radiator",
"n04041544": "radio, wireless",
"n04044716": "radio telescope, radio reflector",
"n04049303": "rain barrel",
"n04065272": "recreational vehicle, RV, R.V.",
"n04067472": "reel",
"n04069434": "reflex camera",
"n04070727": "refrigerator, icebox",
"n04074963": "remote control, remote",
"n04081281": "restaurant, eating house, eating place, eatery",
"n04086273": "revolver, six-gun, six-shooter",
"n04090263": "rifle",
"n04099969": "rocking chair, rocker",
"n04111531": "rotisserie",
"n04116512": "rubber eraser, rubber, pencil eraser",
"n04118538": "rugby ball",
"n04118776": "rule, ruler",
"n04120489": "running shoe",
"n04125021": "safe",
"n04127249": "safety pin",
"n04131690": "saltshaker, salt shaker",
"n04133789": "sandal",
"n04136333": "sarong",
"n04141076": "sax, saxophone",
"n04141327": "scabbard",
"n04141975": "scale, weighing machine",
"n04146614": "school bus",
"n04147183": "schooner",
"n04149813": "scoreboard",
"n04152593": "screen, CRT screen",
"n04153751": "screw",
"n04154565": "screwdriver",
"n04162706": "seat belt, seatbelt",
"n04179913": "sewing machine",
"n04192698": "shield, buckler",
"n04200800": "shoe shop, shoe-shop, shoe store",
"n04201297": "shoji",
"n04204238": "shopping basket",
"n04204347": "shopping cart",
"n04208210": "shovel",
"n04209133": "shower cap",
"n04209239": "shower curtain",
"n04228054": "ski",
"n04229816": "ski mask",
"n04235860": "sleeping bag",
"n04238763": "slide rule, slipstick",
"n04239074": "sliding door",
"n04243546": "slot, one-armed bandit",
"n04251144": "snorkel",
"n04252077": "snowmobile",
"n04252225": "snowplow, snowplough",
"n04254120": "soap dispenser",
"n04254680": "soccer ball",
"n04254777": "sock",
"n04258138": "solar dish, solar collector, solar furnace",
"n04259630": "sombrero",
"n04263257": "soup bowl",
"n04264628": "space bar",
"n04265275": "space heater",
"n04266014": "space shuttle",
"n04270147": "spatula",
"n04273569": "speedboat",
"n04275548": "spider web, spider's web",
"n04277352": "spindle",
"n04285008": "sports car, sport car",
"n04286575": "spotlight, spot",
"n04296562": "stage",
"n04310018": "steam locomotive",
"n04311004": "steel arch bridge",
"n04311174": "steel drum",
"n04317175": "stethoscope",
"n04325704": "stole",
"n04326547": "stone wall",
"n04328186": "stopwatch, stop watch",
"n04330267": "stove",
"n04332243": "strainer",
"n04335435": "streetcar, tram, tramcar, trolley, trolley car",
"n04336792": "stretcher",
"n04344873": "studio couch, day bed",
"n04346328": "stupa, tope",
"n04347754": "submarine, pigboat, sub, U-boat",
"n04350905": "suit, suit of clothes",
"n04355338": "sundial",
"n04355933": "sunglass",
"n04356056": "sunglasses, dark glasses, shades",
"n04357314": "sunscreen, sunblock, sun blocker",
"n04366367": "suspension bridge",
"n04367480": "swab, swob, mop",
"n04370456": "sweatshirt",
"n04371430": "swimming trunks, bathing trunks",
"n04371774": "swing",
"n04372370": "switch, electric switch, electrical switch",
"n04376876": "syringe",
"n04380533": "table lamp",
"n04389033": "tank, army tank, armored combat vehicle, armoured combat vehicle",
"n04392985": "tape player",
"n04398044": "teapot",
"n04399382": "teddy, teddy bear",
"n04404412": "television, television system",
"n04409515": "tennis ball",
"n04417672": "thatch, thatched roof",
"n04418357": "theater curtain, theatre curtain",
"n04423845": "thimble",
"n04428191": "thresher, thrasher, threshing machine",
"n04429376": "throne",
"n04435653": "tile roof",
"n04442312": "toaster",
"n04443257": "tobacco shop, tobacconist shop, tobacconist",
"n04447861": "toilet seat",
"n04456115": "torch",
"n04458633": "totem pole",
"n04461696": "tow truck, tow car, wrecker",
"n04462240": "toyshop",
"n04465501": "tractor",
"n04467665": "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
"n04476259": "tray",
"n04479046": "trench coat",
"n04482393": "tricycle, trike, velocipede",
"n04483307": "trimaran",
"n04485082": "tripod",
"n04486054": "triumphal arch",
"n04487081": "trolleybus, trolley coach, trackless trolley",
"n04487394": "trombone",
"n04493381": "tub, vat",
"n04501370": "turnstile",
"n04505470": "typewriter keyboard",
"n04507155": "umbrella",
"n04509417": "unicycle, monocycle",
"n04515003": "upright, upright piano",
"n04517823": "vacuum, vacuum cleaner",
"n04522168": "vase",
"n04523525": "vault",
"n04525038": "velvet",
"n04525305": "vending machine",
"n04532106": "vestment",
"n04532670": "viaduct",
"n04536866": "violin, fiddle",
"n04540053": "volleyball",
"n04542943": "waffle iron",
"n04548280": "wall clock",
"n04548362": "wallet, billfold, notecase, pocketbook",
"n04550184": "wardrobe, closet, press",
"n04552348": "warplane, military plane",
"n04553703": "washbasin, handbasin, washbowl, lavabo, wash-hand basin",
"n04554684": "washer, automatic washer, washing machine",
"n04557648": "water bottle",
"n04560804": "water jug",
"n04562935": "water tower",
"n04579145": "whiskey jug",
"n04579432": "whistle",
"n04584207": "wig",
"n04589890": "window screen",
"n04590129": "window shade",
"n04591157": "Windsor tie",
"n04591713": "wine bottle",
"n04592741": "wing",
"n04596742": "wok",
"n04597913": "wooden spoon",
"n04599235": "wool, woolen, woollen",
"n04604644": "worm fence, snake fence, snake-rail fence, Virginia fence",
"n04606251": "wreck",
"n04612504": "yawl",
"n04613696": "yurt",
"n06359193": "web site, website, internet site, site",
"n06596364": "comic book",
"n06785654": "crossword puzzle, crossword",
"n06794110": "street sign",
"n06874185": "traffic light, traffic signal, stoplight",
"n07248320": "book jacket, dust cover, dust jacket, dust wrapper",
"n07565083": "menu",
"n07579787": "plate",
"n07583066": "guacamole",
"n07584110": "consomme",
"n07590611": "hot pot, hotpot",
"n07613480": "trifle",
"n07614500": "ice cream, icecream",
"n07615774": "ice lolly, lolly, lollipop, popsicle",
"n07684084": "French loaf",
"n07693725": "bagel, beigel",
"n07695742": "pretzel",
"n07697313": "cheeseburger",
"n07697537": "hotdog, hot dog, red hot",
"n07711569": "mashed potato",
"n07714571": "head cabbage",
"n07714990": "broccoli",
"n07715103": "cauliflower",
"n07716358": "zucchini, courgette",
"n07716906": "spaghetti squash",
"n07717410": "acorn squash",
"n07717556": "butternut squash",
"n07718472": "cucumber, cuke",
"n07718747": "artichoke, globe artichoke",
"n07720875": "bell pepper",
"n07730033": "cardoon",
"n07734744": "mushroom",
"n07742313": "Granny Smith",
"n07745940": "strawberry",
"n07747607": "orange",
"n07749582": "lemon",
"n07753113": "fig",
"n07753275": "pineapple, ananas",
"n07753592": "banana",
"n07754684": "jackfruit, jak, jack",
"n07760859": "custard apple",
"n07768694": "pomegranate",
"n07802026": "hay",
"n07831146": "carbonara",
"n07836838": "chocolate sauce, chocolate syrup",
"n07860988": "dough",
"n07871810": "meat loaf, meatloaf",
"n07873807": "pizza, pizza pie",
"n07875152": "potpie",
"n07880968": "burrito",
"n07892512": "red wine",
"n07920052": "espresso",
"n07930864": "cup",
"n07932039": "eggnog",
"n09193705": "alp",
"n09229709": "bubble",
"n09246464": "cliff, drop, drop-off",
"n09256479": "coral reef",
"n09288635": "geyser",
"n09332890": "lakeside, lakeshore",
"n09399592": "promontory, headland, head, foreland",
"n09421951": "sandbar, sand bar",
"n09428293": "seashore, coast, seacoast, sea-coast",
"n09468604": "valley, vale",
"n09472597": "volcano",
"n09835506": "ballplayer, baseball player",
"n10148035": "groom, bridegroom",
"n10565667": "scuba diver",
"n11879895": "rapeseed",
"n11939491": "daisy",
"n12057211": "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
"n12144580": "corn",
"n12267677": "acorn",
"n12620546": "hip, rose hip, rosehip",
"n12768682": "buckeye, horse chestnut, conker",
"n12985857": "coral fungus",
"n12998815": "agaric",
"n13037406": "gyromitra",
"n13040303": "stinkhorn, carrion fungus",
"n13044778": "earthstar",
"n13052670": "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
"n13054560": "bolete",
"n13133613": "ear, spike, capitulum",
"n15075141": "toilet tissue, toilet paper, bathroom tissue"}