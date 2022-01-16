# ```
# bounding boxes from heatmap:
# https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_find_object.html
# ```
import numpy as np
import os

def assert_folder(folder):
    import os
    if not os.path.exists(folder):
        f_path, f_name = os.path.split(folder)
        if len(f_path)>0:
            assert_folder(f_path)
        os.mkdir(folder)
    return folder

def export_dets_CSV(q_dets, dets_export_fname):
    # q_dets structure: format B, produced in FSD_engine.cast_detections_A2B()
    # q_dets is a list of individual boxes on a single image
    # each list entry is [det_row, cat_ord, cat_name], where det_row=[left, top, right, bottom, score]
    with open(dets_export_fname, 'w') as fid_w:
        # fid_w.write('%s\n' % format('<Cat name>;<score>;<Left>;<Top>;<Right>;<Bottom>;<class number>'))
        for entry in q_dets:
            bbox = entry[0] # [left, top, right, bottom, score]
            cat_ord = entry[1]
            cat_name = entry[2]
            tline = '{0};{1:.6e};{2:.6e};{3:.6e};{4:.6e};{5:.6e};{6}'.format(cat_name, bbox[4], int(bbox[0]), int(bbox[1]), \
                                                             int(bbox[2]), int(bbox[3]), cat_ord)
            fid_w.write('%s\n' % format(tline))

def get_IoU(bb, gt):

    ixmin = np.maximum(gt[0], bb[0])
    iymin = np.maximum(gt[1], bb[1])
    ixmax = np.minimum(gt[2], bb[2])
    iymax = np.minimum(gt[3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (gt[2] - gt[0] + 1.) *
           (gt[3] - gt[1] + 1.) - inters)

    overlap = inters / uni
    return overlap


def get_GT_IoUs(det, bbgt):
    bb = det[:4]
    ixmin = np.maximum(bbgt[:, 0], bb[0])
    iymin = np.maximum(bbgt[:, 1], bb[1])
    ixmax = np.minimum(bbgt[:, 2], bb[2])
    iymax = np.minimum(bbgt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (bbgt[:, 2] - bbgt[:, 0] + 1.) *
           (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def get_GT_IoPs(det, bbgt):
    bb = det[:4]
    ixmin = np.maximum(bbgt[:, 0], bb[0])
    iymin = np.maximum(bbgt[:, 1], bb[1])
    ixmax = np.minimum(bbgt[:, 2], bb[2])
    iymax = np.minimum(bbgt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    pred_area = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.)


    overlaps = inters / pred_area
    return overlaps

def get_GT_IoGs(det, bbgt):
    bb = det[:4]
    ixmin = np.maximum(bbgt[0], bb[0])
    iymin = np.maximum(bbgt[1], bb[1])
    ixmax = np.minimum(bbgt[2], bb[2])
    iymax = np.minimum(bbgt[3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih


    gt_area = (bbgt[2] - bbgt[0] + 1.) * (bbgt[3] - bbgt[1] + 1.)


    overlaps = inters / gt_area
    return overlaps
