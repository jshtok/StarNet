#------------------------------------------------------------------
# StarNet Weakly Supervised Few-Shot object detection
# Copyright (c) 2021 IBM Corp.
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Joseph Shtok, josephs@il.ibm.com, Leonid Karlinsky, leonidka@il.ibm.com, IBM Research AI, Haifa, Israel
#------------------------------------------------------------------

# -*- coding: utf-8 -*-
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
#from models.classification_heads import report_query_images,report_query_images_noGT
from matplotlib import cm
#from tensorboardX import SummaryWriter
from aux_routines.bbox_routines import heatmap2bbox,grid2img
from aux_routines.PerfStats import PerfStats
from aux_routines.data_structures import export_dets_CSV
from aux_routines.bbox_routines import heatmap2bbox_CAM
from utils import convert_batch2vis
from skimage.io import imsave, imread
from utils import set_gpu, Timer, count_accuracy, check_dir, log, get_model, print_memory_usage, one_hot,assert_folder
import shutil

val_phase ='test'
ip_addr = '9.148.203.40'
# -------------------------------------------------------------------------------------------------------------------------



def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'cub':
        from data.cub_bb import Cub, FewShotDataloader
        dataset_train = Cub(phase='train')
        dataset_val = Cub(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'imagenet-loc':
        from data.imagenet_loc import ImageNet, AnnoImageNet, FewShotDataloader
        dataset_train = ImageNet(options, phase='train')
        dataset_val = AnnoImageNet(options, phase=val_phase)
        # dataset_val = ImageNet(options, phase=val_phase)
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    return (dataset_train, dataset_val, data_loader)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--debug_port', default=1327, type=int, help='for debug')
    parser.add_argument('--debug_addr', default=ip_addr, type=str, help='for debug')
    parser.add_argument('--num-epoch', type=int, default=150,                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=5,                             help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=1,                             help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,                             help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=1,                             help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,                             help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=5,                             help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,                             help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,                             help='number of classes in one test (or validation) episode')
    parser.add_argument('--voc_split', type=int, default=1, help='Pascal voc train-test split')
    parser.add_argument('--save-path', default='./experiments/starNet_inloc168_sn_2')
    parser.add_argument('--gpu', default='system_set')
    parser.add_argument('--network', type=str, default='ResNet_star_hi',help='choose which embedding network to use. ProtoNet, R2D2, ResNet') # ProtoNet'
    parser.add_argument('--head', type=str, default='StarNet',                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=4,                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.1,                            help='epsilon of label smoothing')
    parser.add_argument('--resume', type=int, default=1,                        help='resume from last saved model')
    parser.add_argument('--resume_init_model_path', type=str, default='no_such_file',                        help='alternative path to the model to resume from if the last model is not found')
    parser.add_argument('--two_stage', type=int, default=1,                        help='one-stage vs two-stage model')
    parser.add_argument('--normalize', type=int, default=1,                        help='normalize the features or not')
    parser.add_argument('--allow_stage_cross_talk', type=int, default=0,                        help='will there be gradients flowing from the second stage to the first stage using the bp_masks')
    parser.add_argument('--stage2_mode', type=int, default=1,                        help='0 = without attention, 1 = query attention only, '
                                                                                          '2 = support attention only, 3 = both attentions')
    parser.add_argument('--vote_prob_nrm_type', type=int, default=1,                        help='0 = without normalization, 1 = sum, 2 = max, '
                                                                                                 '3 = sum current, 4 = max current')
    parser.add_argument('--base_lr', type=float, default=0.01, help='starting LR')
    parser.add_argument('--stages_lr_factor', type=float, default=1.0, help='relative LR factor for scheduler')
    parser.add_argument('--split_head_scales', type=int, default=1,                        help='0 = one scale for all, 1 = separate scale per stage')
    parser.add_argument('--stage2_cosine', type=int, default=0,                        help='0 = L2 sim, 1 = cosine sim')
    parser.add_argument('--support_comb_type', type=str, default='avg',                        help='avg / max')
    parser.add_argument('--stage2_type', type=str, default='proto',                        help='proto / star')
    parser.add_argument('--multiple_hypotheses_perc', type=float, default=0.5,                       help='NMS parameter, determines where to stop '
                                                                                                        'supressing, if 1.0 no multi-hypothese are used')
    parser.add_argument('--scheduler_regime',type=int,default=0)
    parser.add_argument('--s1', type=str, default='[[1.0, 20], [0.06, 40], [0.012, 50], 0.0024]',    help='scheduler for backbone1 or for all if no two stages are invoked')
    parser.add_argument('--s2', type=str, default='',                      help='scheduler for backbone2')
    parser.add_argument('--s3', type=str, default='',                       help='scheduler for head')
    parser.add_argument('--test_classes_list_path', type=str, default='None', help='alternative list of test categories (use for validation')
    parser.add_argument('--obj_portion', type=float, default=0.0, help='crop image to have the object (in relevant class) to occupy this portion of the image')
    parser.add_argument('--obj_select_mode', type=int, default=11, help='0/10 - take minimal image containing all the instances of episode class; 1/11 - take largest box of episode class.')
    parser.add_argument('--mean_clr', type=str, default='[120.39586422, 115.59361427, 104.54012653]',
                        help='data mean')
    parser.add_argument('--std_clr', type=str, default='[70.68188272, 68.27635443, 72.54505529]',
                        help='data std')
    parser.add_argument('--data_gen_folder', type=str, default='./exp_data')

    #------------------------------
    parser.add_argument('--image_res', type=int, default=84, help='image resolution')
    parser.add_argument('--bp_sigma', type=float, default=2.0, help='bp sigma')
    parser.add_argument('--val_iters', type=int, default=0, help='')
    #parser.add_argument('--train-iters', type=int, default=0, help='') replaced by do_train flag
    parser.add_argument('--do_train', action='store_true', help='')
    parser.add_argument('--pad_mode', type=str, default='constant',    help='padding type for image res conversion')
    parser.add_argument('--recompute_dataset_dicts', type=int, default=0, help='')
    parser.add_argument('--crop_style', type=int, default=1, help='imagenet crop style')
    parser.add_argument('--voting_kernel_sigma', type=float, default=1, help='kernel sigma for voting step')
    # --- detection ------------------------------------------------
    parser.add_argument('--IoU_thresh', type=str, default='[0.3,0.5]', help='IoU overlap for detection performance measurement')
    parser.add_argument('--bbox_method', type=int, default=2, help='method for bbox. 0 -trivial box of image size, '
                                                                   '1 - use skimage.measure.regionprops (heatmat2bbox routine), 2 - use CAM, 3 - use GrabCut')
    parser.add_argument('--fg_measure', type=str, default='[0]', help='Foreground geometric condition. 0 - IoU, 1 - IoP (intersection over '
                                                                       'prediction)')
    parser.add_argument('--gen_bbox', type=int, default=0)
    parser.add_argument('--hmap_thresh', type=float, default=0.75,help = 'backprojectaion heatmap threshold for box prediction')
    parser.add_argument('--bb_process', type=int, default=1,           help='heatmap2bbox process type. 0 - fixed threshold, 1 - hmap_thresh '
                                                                            'percentage of the top')
    parser.add_argument('--bb_erode', type=int, default=1,   help='thresholded bp heatmap erosion param')
    parser.add_argument('--bb_dilate', type=int, default=3,  help='thresholded bp heatmap erosion param')
    parser.add_argument('--work_folder', type=str, default='None', help='use this name for test folder')
    parser.add_argument('--bbox_CAM_par', type=str, default='[0.8, 0.2, 40, 130, 150, 1]', help='parameters of heatmap2bbox_CAM ')  # '[0.6,0.4,20,100,110,1]'
    parser.add_argument('--final_result_to', type=str, default='None', help='print final result to this file')
    parser.add_argument('--bbox_scale', type=float, default=0.85, help='print final result to this file')
    parser.add_argument('--single_class_per_im', type=float, default=0, help='use just the max score - sinlge class')
    parser.add_argument('--resume_model_name', type=str, default='None', help='load from this model')
    parser.add_argument('--self_sim', type=int, default=1, help='perform self-similarity heatmap expansion')
    parser.add_argument('--self_sim_display', type=int, default=0, help='show self_sim difference')

    parser.add_argument('--ssim_hmap_thresh', type=float, default=0.75, help='')
    parser.add_argument('--ssim_second_thresh', type=float, default=1e-7, help='')
    parser.add_argument('--n_ssim_points', type=int, default=15, help='')
    parser.add_argument('--ssim_add_value', type=float, default=3.0, help='')

    opt = parser.parse_args()
    if opt.scheduler_regime == 0:
        opt.s1 =   "[[1.0, 40], [0.1, 70], [0.01, 130], 0.001]"
        opt.s2 =   "[[1.0, 90], [0.1, 110], [0.01, 130], 0.001]"
        opt.s3 =   "[[1.0, 90], [0.1, 110], [0.01, 130], 0.001]"
    elif opt.scheduler_regime == 1:
        opt.s1 =   "[[1.0, 20], [0.1, 40], [0.01, 60], 0.001]"
        opt.s2 =   "[[1.0, 30], [0.1, 60], [0.01, 90], 0.001]"
        opt.s3 =   "[[1.0, 40], [0.1, 80], [0.01, 130], 0.001]"

    opt.s1 = eval(opt.s1)
    if len(opt.s2) > 0:
        opt.s2 = eval(opt.s2)
    else:
        opt.s2 = opt.s1
    if len(opt.s3) > 0:
        opt.s3 = eval(opt.s3)
    else:
        opt.s3 = opt.s1
    opt.bbox_CAM_par = eval(opt.bbox_CAM_par)
    opt.IoU_thresh = eval(opt.IoU_thresh)
    opt.fg_measure = eval(opt.fg_measure)
    return opt

def self_sim_correction(hmap,img_vis,prox_query,test_batch_path,q,opt,do_display):
    # - self-similarity algo -------------
    ssim_hmap_thresh = opt.ssim_hmap_thresh # threshold for finding the seed points for similarity search
    ssim_second_thresh = opt.ssim_second_thresh # threshold for similarity - which similar points to use
    n_ssim_points = opt.n_ssim_points
    ssim_add_value = opt.ssim_add_value
    def blend_hmap(img_vis,hmap_n)   :
        hmap_nv = cv2.resize(hmap_n, (img_vis.shape[0], img_vis.shape[1]))
        hmap_3 = 255 * cm.jet(hmap_nv / hmap_n.max())[:, :, :-1]
        hmap_3 = hmap_3[:, :, [2, 1, 0]]
        blend = np.uint8(0.65 * img_vis + 0.35 * hmap_3)
        return blend

    hmap_n = np.asarray(hmap); n = hmap_n.shape[0]
    hmap_ns = np.copy(hmap_n)  # leave hmap_n for backup

    high_inds = np.where(hmap > ssim_hmap_thresh)
    x = high_inds[1].tolist()
    y = high_inds[0].tolist()
    if do_display:
        # print('hmap_n:')
        # print((hmap_n > thresh).astype(int))
        print('hmap_n: {} high_ind points retrieved'.format(len(high_inds[0])))

    for iidx in range(min(len(x), n_ssim_points)):
        xx = x[iidx]; yy = y[iidx]
        local_sim_map = prox_query[yy * n + xx].view(n, n) # prox_query = prox_query_set[q]
        local_sim_map = np.asarray(local_sim_map.detach().cpu())
        if do_display:
            for thresh in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
                print('local sim map {}'.format(thresh))
                print((local_sim_map >thresh).astype(int))

        char_map = (np.asarray(local_sim_map) > ssim_second_thresh).astype(int)
        hmap_ns = hmap_ns + ssim_add_value * char_map
        if np.sum(hmap_ns >= 1) / hmap_ns.size > 0.3:
            break
    ccut = 1.0
    hmap_ns[np.where(hmap_ns >= ccut)] = ccut

    if do_display:
        blend1 = blend_hmap(img_vis, hmap_n)
        blend2 = blend_hmap(img_vis,hmap_ns)
        cv2.imwrite(os.path.join(test_batch_path,'ssim_comp_{}.jpg'.format(q)), np.concatenate((blend1, blend2), axis=1))
        #imsave(os.path.join(test_batch_path,'ssim_comp_{}.jpg'.format(q)), np.concatenate((blend1, blend2), axis=1))
        a = 1
    # bpp = torch.Tensor(cm.jet(hmap / hmap.max())[:, :, :-1]).permute(2, 0, 1)
    hmap = torch.from_numpy(hmap_ns)
    return hmap

def setup_debug(args):
    if args.debug:
        import pydevd_pycharm
        pydevd_pycharm.settrace(args.debug_addr, port=args.debug_port, stdoutToServer=True, stderrToServer=True, suspend=False)

def switch_to_eval(embedding_net, cls_head):
    if type(embedding_net) is list:
        for net in embedding_net:
            net.eval()
        cls_head.eval()
    else:
        _, _ = [x.eval() for x in (embedding_net, cls_head)]
    return cls_head

def process_batch_data(batch,embedding_net_st1,embedding_net_st2, opt):
    try:
        data_support, labels_support, _, data_query, labels_query, boxes_query, _, _ = [x.cuda() for x in batch]
    except:
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
        boxes_query = None

    nClasses = labels_support.shape[1]
    cat_ords = list(range(nClasses))
    test_n_support = opt.test_way * opt.val_shot
    test_n_query = opt.test_way * opt.val_query

    emb_support = embedding_net_st1(data_support.reshape([-1] + list(data_support.shape[-3:])))
    if 'Star' in opt.head:
        emb_support = emb_support.reshape([1, test_n_support] + list(emb_support.shape[-3:]))
    else:
        emb_support = emb_support.reshape(1, test_n_support, -1)

    emb_query = embedding_net_st1(data_query.reshape([-1] + list(data_query.shape[-3:])))
    if 'Star' in opt.head:
        emb_query = emb_query.reshape([1, test_n_query] + list(emb_query.shape[-3:]))
    else:
        emb_query = emb_query.reshape(1, test_n_query, -1)

    s2_emb_support, s2_emb_query = None, None
    if embedding_net_st2 is not None:
        s2_emb_support = embedding_net_st2(data_support.reshape([-1] + list(data_support.shape[-3:])))
        if 'Star' in opt.head:
            s2_emb_support = s2_emb_support.reshape(
                [1, test_n_support] + list(s2_emb_support.shape[-3:]))
        else:
            s2_emb_support = s2_emb_support.reshape(1, test_n_support, -1)

        s2_emb_query = embedding_net_st2(data_query.reshape([-1] + list(data_query.shape[-3:])))
        if 'Star' in opt.head:
            s2_emb_query = s2_emb_query.reshape([1, test_n_query] + list(s2_emb_query.shape[-3:]))
        else:
            s2_emb_query = s2_emb_query.reshape(1, test_n_query, -1)
    return   boxes_query, emb_query, emb_support,data_query, data_support, labels_query,labels_support, s2_emb_support, s2_emb_query, cat_ords

def produce_detections(emb_query, bp_query, logit_query, data_query, boxes_query, labels_query, prox_query_set, dets_folder, dataset_val, data_support,
                       class_indices, opt, display_dev=False):

    heatmaps = bp_query.detach().cpu()
    logit_query_num = logit_query.detach().cpu()
    bboxes_epi = []
    gt_boxes_set = []
    gt_classes_set = []
    pred_classses_epi = []
    for b in range(logit_query.shape[0]):  # episodes
        for q in range(logit_query.shape[1]):  # queries
            img_dims = data_query[b, q].size()
            hmap_dims = heatmaps[b, q, 0].size()
            q_dets_B = []
            pred_class = int(torch.argmax(logit_query_num[b, q]))  #

            # pred_classes = [p for p in range(np.asarray(logit_query_num[b, q]).shape[0])]  # select the best class (assuming single class per image)
            # pred_classses_set.append(pred_classes)
            hmaps = heatmaps[b, q, :].cpu()
            # hmap = hmap / hmap.max()
            hmaps = np.asarray(hmaps) / np.expand_dims(
                np.expand_dims(np.max(np.max(np.asarray(hmaps), axis=1), axis=1), axis=1), axis=1)

            img_vis = convert_batch2vis(data_query[b, q].cpu(), 255 * np.asarray(dataset_val.mean_pix),
                                        255 * np.asarray(dataset_val.std_pix))
            pred_bboxes = []
            pred_classes = []

            if opt.bbox_method == 1:
                for cls_idx, hmap in enumerate(hmaps):
                    bboxes_gr = heatmap2bbox(hmap, opt.hmap_thresh, opt.bb_process,
                                             opt.bb_erode, opt.bb_dilate)  # [top, left, bottom, right]
                    for bbox in bboxes_gr:
                        pred_bboxes.append(grid2img(bbox, hmap_dims[1], img_dims[1]))
                        pred_classes.append(cls_idx)

            elif opt.bbox_method == 2:  # CAM algoritm
                save_file_prefix = os.path.join(dets_folder, 'hm2_{0}_{1}_'.format(b, q))

                for cls_idx, hmap in enumerate(hmaps):
                    if opt.self_sim == 1:
                        hmap_ss = self_sim_correction(hmap, img_vis, prox_query_set[q], dets_folder, q, opt,
                                                      cls_idx == pred_class and opt.self_sim_display)
                    else:
                        hmap_ss = hmap
                    bboxes_cam = heatmap2bbox_CAM(hmap_ss, img_vis, save_file_prefix,
                                                  fuse_portions=opt.bbox_CAM_par[0:2],
                                                  bbox_threshold=opt.bbox_CAM_par[2:5])
                    if len(bboxes_cam) <= 1:  # failsafe. return to original heatmap
                        bboxes_cam = heatmap2bbox_CAM(hmap, img_vis, save_file_prefix,
                                                      fuse_portions=opt.bbox_CAM_par[0:2],
                                                      bbox_threshold=opt.bbox_CAM_par[2:5])
                    else:
                        hmaps[cls_idx] = hmap_ss

                    bbox = bboxes_cam[opt.bbox_CAM_par[5]].tolist()
                    if pred_class == cls_idx:
                        bboxes_epi.append([bbox])
                        pred_classses_epi.append([pred_class])

                    if opt.single_class_per_im:
                        if pred_class == cls_idx:
                            pred_bboxes.append(bbox)
                            pred_classes.append(cls_idx)
                    else:
                        pred_bboxes.append(bbox)
                        pred_classes.append(cls_idx)

            heatmaps[b, q, :] = torch.from_numpy(hmaps)

            if opt.bbox_scale != 1.0:
                for ib, bbox in enumerate(pred_bboxes):
                    bbox_22 = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
                    c = np.mean(bbox_22, axis=0)
                    bbox_22 = bbox_22 - c
                    bbox_22 = bbox_22 * opt.bbox_scale
                    bbox_22 = bbox_22 = bbox_22 + c
                    bbox = bbox_22.flatten().tolist()
                    bbox = [int(b) for b in bbox]
                    pred_bboxes[ib] = bbox

            detsPath = os.path.join(dets_folder, 'dets_{0}_{1}.txt'.format(b, q))

            for idx, bbox in enumerate(pred_bboxes):
                entry = [[] for _ in range(3)]
                entry[0] = bbox + [
                    float(logit_query_num[b, q, idx])]  # box corners + score value
                entry[1] = str(idx)
                entry[2] = 'object'  # TODO: map class to object name
                q_dets_B.append(entry)

            export_dets_CSV(q_dets_B, detsPath)

            gt_boxes = np.asarray(
                [np.asarray(bbox.cpu()) for bbox in boxes_query[0][q] if torch.max(bbox) > 0])
            gt_classes = [str(labels_query.cpu().tolist()[b][q]) for bbox in gt_boxes]
            gt_boxes_set.append(gt_boxes)
            gt_classes_set.append(gt_classes)
            if display_dev:
                from aux_routines.show_boxes import show_detsB_gt_boxes

                img = data_query[b, q]
                img_vis = convert_batch2vis(img.cpu(), 255 * np.asarray(dataset_val.mean_pix),
                                            255 * np.asarray(dataset_val.std_pix))
                show_detsB_gt_boxes(img_vis[:, :, [2, 1, 0]], q_dets_B, gt_boxes, gt_classes,
                                    save_file_path=os.path.join(dets_folder,
                                                                'dets_{0}_{1}.jpg'.format(b, q)))
        # display -------------------------------------------------------------

        b, nQ, ch, y, x = emb_query.shape
        if False:
            report_query_images_noGT(
                inputs=(data_support, data_query),
                mean_clr=opt.mean_clr,
                std_clr=opt.std_clr,
                c=class_indices,
                bboxes=bboxes_epi,
                bp_query=bp_query,
                pred_classses_set=pred_classses_epi,
                gt_boxes_set=gt_boxes_set,
                gt_classes_set=gt_classes_set,
                nQ=nQ,
                img_save_path=dets_folder
            )

def main():
    opt = get_opt()
    setup_debug(opt)

    max_val_acc = 0.0
    first_epoch = 1
    tb_report_freq = 100

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)
    if opt.do_train:
        nrm = [x for x in dataset_train.transform.transforms if isinstance(x, transforms.Normalize)][0]
    else:
        nrm = [x for x in dataset_val.transform.transforms if isinstance(x, transforms.Normalize)][0]

    opt.mean_clr = [x * 255.0 for x in nrm.mean]
    opt.std_clr = [x * 255.0 for x in nrm.std]

    if opt.do_train:
        dloader_train = data_loader(
            dataset=dataset_train,
            nKnovel=opt.train_way,
            nKbase=0,
            nExemplars=opt.train_shot,  # num training examples per novel category
            nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
            nTestBase=0,  # num test examples for all the base categories
            batch_size=opt.episodes_per_batch,
            num_workers=4,
            epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch
        )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    log_dir = os.path.join(opt.save_path, 'tb_logs')
    check_dir(log_dir)

    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    work_folder = curr_time

    assert_folder(os.path.join(opt.save_path, work_folder))
    if opt.do_train:
        log_file_path = os.path.join(opt.save_path, "train_log.txt")
    else:
        log_file_path = os.path.join(opt.save_path, work_folder, 'test_{}.log'.format(curr_time))
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    learning_rate = opt.base_lr

    if type(embedding_net) is list:
        pGroups = [{'params': net.parameters()} for net in embedding_net]
        optimizer = torch.optim.SGD(
            pGroups + [{'params': cls_head.parameters()}],
            lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True
        )
        # f = opt.stages_lr_factor  # 1e-4
        lambda_epoch = [
            lambda e: opt.s1[0][0] if e < opt.s1[0][1] else (opt.s1[1][0] if e < opt.s1[1][1] else opt.s1[2][0] if e < opt.s1[2][1] else opt.s1[3]),
            lambda e: opt.s2[0][0] if e < opt.s2[0][1] else (opt.s2[1][0] if e < opt.s2[1][1] else opt.s2[2][0] if e < opt.s2[2][1] else opt.s2[3]),
            lambda e: opt.s3[0][0] if e < opt.s3[0][1] else (opt.s3[1][0] if e < opt.s3[1][1] else opt.s3[2][0] if e < opt.s3[2][1] else opt.s3[3]),
        ]
    else:
        lambda_epoch = lambda e: opt.s1[0][0] if e < opt.s1[0][1] else (opt.s1[1][0] if e < opt.s1[1][1] else opt.s1[2][0] if e < opt.s1[2][1] else opt.s1[3])
        optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                     {'params': cls_head.parameters()}], lr=learning_rate, momentum=0.9, \
                                    weight_decay=5e-4, nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)


    if opt.resume:
        ignore_states = False
        strict_load_state = True
        if not opt.resume_model_name == 'None':
            last_epoch_path = os.path.join(opt.save_path, opt.resume_model_name)
        else:
            last_epoch_path = os.path.join(opt.save_path, 'last_epoch.pth')

        if not os.path.isfile(last_epoch_path):
            last_epoch_path = opt.resume_init_model_path
            ignore_states = True
            strict_load_state = False
            print(f'Warning! model not loaded - path {last_epoch_path} doesnt`t exist! ------------------------->')
        else:
            log(log_file_path, 'Resuming from the model loaded from {}'.format(last_epoch_path))
            model_data = torch.load(last_epoch_path)
            if type(embedding_net) is list:
                loaded = model_data['embedding']
                if type(loaded) is list:
                    for iNet, net in enumerate(embedding_net):
                        net.load_state_dict(loaded[iNet])
                else:
                    embedding_net[0].load_state_dict(loaded)
            else:
                embedding_net.load_state_dict(model_data['embedding'])
            cls_head.load_state_dict(model_data['head'], strict=strict_load_state)
            if not ignore_states:
                max_val_acc = model_data['max_val_acc']
                optimizer.load_state_dict(model_data['optimizer'])
                lr_scheduler.load_state_dict(model_data['lr_scheduler'])
                first_epoch = model_data['epoch'] + 1

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    perfStats_global_set = [[] for _ in range(len(opt.fg_measure) * len(opt.IoU_thresh))]
    N_f = len(opt.fg_measure)
    for idx_t, thr in enumerate(opt.IoU_thresh):
        for idx_f, fg_measure in enumerate(opt.fg_measure):
            perfStats_global_set[idx_t * N_f + idx_f] = PerfStats()

    val_iter_cnt = -1

    for epoch in range(first_epoch, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for iG, param_group in enumerate(optimizer.param_groups):
            epoch_learning_rate = param_group['lr']
            log(log_file_path, 'Train Epoch: {}\tGroup #{}\tLearning Rate: {:.7f}'.format(
                epoch, iG, epoch_learning_rate))

        # may be helpful to catch memory leaks
        print_memory_usage(log_file_path)

        if type(embedding_net) is list:
            for net in embedding_net:
                net.train()
            cls_head.train()
        else:
            _, _ = [x.train() for x in (embedding_net, cls_head)]

        train_accuracies = []
        train_losses = []
        train_accuracies_parts = []
        train_losses_parts = []

        if type(embedding_net) is list:
            embedding_net_st1 = embedding_net[0]
            embedding_net_st2 = embedding_net[1]  # TODO: support more than one
        else:
            embedding_net_st1 = embedding_net
            embedding_net_st2 = None

        # --- training --------------------------------------------------------------------
        if opt.do_train:
            for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
                if len(batch) == 6:
                    data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
                elif len(batch) == 8:
                    data_support, labels_support, boxes_support, data_query, labels_query, boxes_query, _, _ = [x.cuda() for x in batch]

                train_n_support = opt.train_way * opt.train_shot
                train_n_query = opt.train_way * opt.train_query

                emb_support = embedding_net_st1(data_support.reshape([-1] + list(data_support.shape[-3:])))
                if 'Star' in opt.head:
                    emb_support = emb_support.reshape([opt.episodes_per_batch, train_n_support] + list(emb_support.shape[-3:]))
                else:
                    emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
                emb_query = embedding_net_st1(data_query.reshape([-1] + list(data_query.shape[-3:])))
                if 'Star' in opt.head:
                    emb_query = emb_query.reshape([opt.episodes_per_batch, train_n_query] + list(emb_query.shape[-3:]))
                else:
                    emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

                s2_emb_support, s2_emb_query = None, None
                if embedding_net_st2 is not None:
                    s2_emb_support = embedding_net_st2(data_support.reshape([-1] + list(data_support.shape[-3:])))
                    if 'Star' in opt.head:
                        s2_emb_support = s2_emb_support.reshape(
                            [opt.episodes_per_batch, train_n_support] + list(s2_emb_support.shape[-3:]))
                    else:
                        s2_emb_support = s2_emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

                    s2_emb_query = embedding_net_st2(data_query.reshape([-1] + list(data_query.shape[-3:])))
                    if 'Star' in opt.head:
                        s2_emb_query = s2_emb_query.reshape([opt.episodes_per_batch, train_n_query] + list(s2_emb_query.shape[-3:]))
                    else:
                        s2_emb_query = s2_emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

                if i % tb_report_freq == 1:
                    logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot, s2_query=s2_emb_query, s2_support=s2_emb_support,
                                           inputs=(data_support, data_query), tb_prefix='train_', opt=opt)
                else:
                    logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot, s2_query=s2_emb_query, s2_support=s2_emb_support, opt=opt)

                if type(logit_query) is tuple:
                    bp_query = logit_query[1]
                    logit_query = logit_query[0]
                else:
                    bp_query = None

                if type(logit_query) is not list:
                    logit_query = [logit_query]

                smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

                loss = 0.0
                loss_parts = [0.0] * len(logit_query)
                for iQ, lq in enumerate(logit_query):
                    log_prb = F.log_softmax(lq.reshape(-1, opt.train_way), dim=1)
                    loss_ = -(smoothed_one_hot * log_prb).sum(dim=1)
                    loss_parts[iQ] = loss_.mean()
                    loss += loss_parts[iQ]

                probs_query = [F.softmax(lq, dim=2) for lq in logit_query]
                logit_query = torch.stack(probs_query, dim=0).sum(dim=0) / float(len(logit_query))

                acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
                acc_parts = [
                    count_accuracy(pq.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    for pq in probs_query
                ]

                train_accuracies.append(acc.item())
                train_losses.append(loss.item())
                train_accuracies_parts.append([x.item() for x in acc_parts])
                train_losses_parts.append([x.item() for x in loss_parts])

                if (i % 100 == 0):
                    train_acc_avg = np.mean(np.array(train_accuracies))
                    loss_avg = np.mean(np.array(train_losses))
                    train_acc_avg_parts = np.mean(np.array(train_accuracies_parts), axis=0)
                    loss_avg_parts = np.mean(np.array(train_losses_parts), axis=0)
                    log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f} ({:.4f}, {})\tAccuracy: {:.2f}% ({:.2f}%, {})'.format(
                        epoch, i, len(dloader_train), loss_avg, loss.item(), loss_avg_parts, train_acc_avg, acc, train_acc_avg_parts))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if type(embedding_net) is list:
                    ensd = [net.state_dict() for net in embedding_net]
                else:
                    ensd = embedding_net.state_dict()

                curr_epoch_path = os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch))
                last_epoch_path = os.path.join(opt.save_path, 'last_epoch.pth')
                last_epoch_bkup_path = os.path.join(opt.save_path, 'last_epoch_bkup.pth')


            if opt.do_train and epoch % opt.save_epoch == 0:
                torch.save(
                    {'embedding': ensd, 'head': cls_head.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'max_val_acc': max_val_acc,
                     'opt': opt}, curr_epoch_path)

            if os.path.exists(last_epoch_path):
                shutil.copy2(last_epoch_path, last_epoch_bkup_path) # in case the environment fails on save, losing results since last opt.save_epoch
            torch.save(
                {'embedding': ensd, 'head': cls_head.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'max_val_acc': max_val_acc}, \
                last_epoch_path)

        # Evaluate on the validation split -------------------------------------------
        if opt.val_iters == 0:
            continue # no test
        cls_head = switch_to_eval(embedding_net, cls_head)
        val_accuracies = []; val_losses = []; val_accuracies_parts = [];  val_losses_parts = []

        for i, batch in enumerate(dloader_val(epoch), 1):
            val_iter_cnt += 1
            if opt.val_iters+1 == i:
                break
            dets_folder = assert_folder(os.path.join(opt.save_path, work_folder, 'dets_' + str(i)))

            boxes_query, emb_query, emb_support,data_query, data_support,labels_query,labels_support,  s2_emb_support,s2_emb_query, \
                cat_ords = process_batch_data(batch, embedding_net_st1,embedding_net_st2, opt)
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, s2_query=s2_emb_query, s2_support=s2_emb_support, tb=tb,
                                   inputs=(data_support, data_query), tb_prefix='val_', opt=opt, img_save_path=dets_folder)
            if type(logit_query) is tuple:
                prox_query_set = logit_query[3]
                class_indices = logit_query[2]
                bp_query = logit_query[1]
                logit_query = logit_query[0]
            else:
                bp_query = None

            if type(logit_query) is not list:
                logit_query = [logit_query]

            loss = 0.0  # compute validation loss for evaluation -----------
            loss_parts = [0.0] * len(logit_query)
            for iQ, lq in enumerate(logit_query):
                loss_parts[iQ] = x_entropy(lq.reshape(-1, opt.test_way), labels_query.reshape(-1))
                loss += loss_parts[iQ]

            probs_query = [F.softmax(lq, dim=2) for lq in logit_query]
            logit_query = torch.stack(probs_query, dim=0).sum(dim=0) / float(len(logit_query))

            if not boxes_query is None:
                produce_detections(emb_query, bp_query, logit_query, data_query, boxes_query, labels_query, prox_query_set, dets_folder,
                                   dataset_val, data_support, class_indices, opt, opt.verbose)

                # GT evaluation -------------------------------------------------------------
                boxes = [np.asarray(b) for b in boxes_query.cpu().squeeze(0)]
                labels = labels_query.cpu().tolist()[0]
                gt_query = {'boxes': boxes, 'gt_classes': labels}

                for idx_t, thr in enumerate(opt.IoU_thresh):
                    for idx_f, fg_measure in enumerate(opt.fg_measure):
                        perfStats_global_set[idx_t * N_f + idx_f].add_base_stats(dets_folder, cat_ords, gt_query, thr, fg_measure)
                        if fg_measure ==0:
                            print_prefix = f'stats  IoU={thr}   epi {i}: '
                        elif fg_measure ==1:
                            print_prefix = f'stats  IoP={thr} epi {i}: '

                        perf_string, perf_data = perfStats_global_set[idx_t * N_f + idx_f].print_perf(prefix=print_prefix)
                        perf_string += ' curr.mean IoG: {:.3f}'.format(np.mean(np.asarray(perf_data[3])))
                        log(log_file_path, perf_string)

            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc_parts = [
                count_accuracy(pq.reshape(-1, opt.test_way), labels_query.reshape(-1))
                for pq in probs_query
            ]

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            val_accuracies_parts.append([x.item() for x in acc_parts])
            val_losses_parts.append([x.item() for x in loss_parts])


            del logit_query
            del emb_support
            del emb_query
            del s2_emb_query
            del s2_emb_support
            del loss_parts
            del loss
            del probs_query
            del acc
            del acc_parts
            del bp_query

        if not opt.final_result_to == 'None':
            log(opt.final_result_to, ' ')
            log(opt.final_result_to, 'final res for {}'.format(opt))
            log(opt.final_result_to, perf_string)

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        val_acc_avg_parts = np.mean(np.array(val_accuracies_parts), axis=0)
        loss_avg_parts = np.mean(np.array(val_losses_parts), axis=0)



        if opt.val_iters > 0:
            if val_acc_avg > max_val_acc:
                max_val_acc = val_acc_avg
                if opt.do_train:
                    torch.save({'embedding': ensd, 'head': cls_head.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch,
                                'max_val_acc': max_val_acc, 'opt': opt}, \
                               os.path.join(opt.save_path, 'best_model.pth'))
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f} ({})\tAccuracy: {:.2f} ± {:.2f} % ({}) (Best)' \
                    .format(epoch, val_loss_avg, loss_avg_parts, val_acc_avg, val_acc_ci95, val_acc_avg_parts))
            else:
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f} ({})\tAccuracy: {:.2f} ± {:.2f} % ({})' \
                    .format(epoch, val_loss_avg, loss_avg_parts, val_acc_avg, val_acc_ci95, val_acc_avg_parts))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))


if __name__ == '__main__':
    main()
