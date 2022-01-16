import os
import numpy as np

def index_roidb_ni(roidb):
    roidb_ni = {}
    for ii, entry in enumerate(roidb):
        image_path = entry['image']
        im_path, im_name = os.path.split(image_path)
        roidb_ni[im_name] = {'boxes':entry['boxes'],'gt_classes':entry['gt_classes'],'gt_names':entry['gt_names']}
    return roidb_ni

def img_dets_CSV_2_A(dets_fname,cat_ords,cand_ver=1):
    # convets detections format from C to A for perf eval.
    q_dets = [ np.zeros((0,5)) for ord in cat_ords]
    cat_names = [ [] for ord in cat_ords]
    with open(dets_fname, 'r') as fid:
        dets_lines = [x.strip() for x in fid.readlines()]
    for det in dets_lines:
        if cand_ver==1:
            fields = det.split(';')
            cat_name = fields[0]
            cat_ord = int(fields[6])
            search_list = np.where(np.asarray(cat_ords) == cat_ord)[0]
            if search_list.shape[0]>0: # index is found
                cat_idx = np.where(np.asarray(cat_ords) == cat_ord)[0][0]
                #det_box = np.expand_dims(np.array([int(fields[2]),int(fields[3]),int(fields[4]),int(fields[5]),float(fields[1])]),axis=0)
                det_box = np.expand_dims(np.array([float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]), float(fields[1])]), axis=0)
                q_dets[cat_idx] = np.concatenate((q_dets[cat_idx],det_box),axis=0) if len(q_dets[cat_idx])>0 else det_box
                cat_names[cat_idx] = cat_name
            else:
                a = 1
                #print('cat_ord {0} was not found in the cat_ords'.format(cat_ord))
        if cand_ver==2:
            print('img_dets_CSV_2_A,cand_ver=2: Not implemented')
            return None,None

    return q_dets,cat_names

def detName_2_imgName(detName):
    imgName = detName[5:-4]
    return imgName

