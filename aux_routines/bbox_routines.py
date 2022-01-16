import numpy as np
from skimage import data, util
from skimage.measure import label, regionprops
import scipy.ndimage
import cv2
import os
def binbask2bbox(bin_mask, bb_erode=3,bb_dilate=5):
    bin_mask = scipy.ndimage.binary_erosion(bin_mask, np.ones((bb_erode, bb_erode))).astype(bin_mask.dtype)
    bin_mask = scipy.ndimage.binary_dilation(bin_mask, np.ones((bb_dilate, bb_dilate))).astype(bin_mask.dtype)
    label_img = label(bin_mask, connectivity=bin_mask.ndim)
    if np.max(label_img)==0:
        print('empty mask obtained --------------')
        return [0,0,1,1]
    props = regionprops(label_img)
    bbox = list(props[0].bbox)
    re_box = [bbox[1],bbox[0],bbox[3],bbox[2]]
    re_box[2] = min(re_box[2],bin_mask.shape[0]-1)
    re_box[3] = min(re_box[3], bin_mask.shape[1] - 1)

    return re_box

def heatmap2bbox(heatmap,thresh, b_process=0, bb_erode=3,bb_dilate=5):

    if b_process==0: # fixed threshold
        bin_mask = util.img_as_ubyte(heatmap) > 255 * thresh
    elif b_process==1: # top 10 percent
        re_hmap = util.img_as_ubyte(heatmap)
        #s_values = np.sort(np.reshape(re_hmap,-1))
        thr = int(np.max(re_hmap*thresh))
        bin_mask = np.where( re_hmap > thr,1,0)

    bin_mask_er = scipy.ndimage.binary_erosion(bin_mask, np.ones((bb_erode, bb_erode))).astype(bin_mask.dtype)
    if np.max(bin_mask_er)==0:
        bin_mask_er = bin_mask
    bin_mask = scipy.ndimage.binary_dilation(bin_mask, np.ones((bb_dilate, bb_dilate))).astype(bin_mask.dtype)
    label_img = label(bin_mask, connectivity=bin_mask.ndim)
    if np.max(label_img)==0:
        print('threshold above max --------------')
        return [[0,0,label_img.shape[0]-1,label_img.shape[1]-1]]
    props = regionprops(label_img)
    bboxes = []
    for pr in props:
        bbox = list(pr.bbox)
        re_box = [bbox[1],bbox[0],bbox[3],bbox[2]]
        re_box[2] = min(re_box[2],heatmap.size()[0]-1)
        re_box[3] = min(re_box[3], heatmap.size()[1] - 1)
        bboxes.append(re_box)
    return bboxes

def grid2img(bbox,hmap_dim,img_dim):
    sc = float(img_dim-1)/float(hmap_dim-1)
    return [int(sc *i) for i in bbox] # for 21 grid, 168 image




def heatmap2bbox_CAM(hmap,img,save_file_prefix, fuse_portions=[0.6,0.4],bbox_threshold = [20, 100, 110]):

    from bbox_generator import gen_bbox_img
    #img = np.uint8(255 * img).transpose((1,2,0)) # to [width, height, 3]
    height, width, _ = img.shape
    hmap_255 = np.uint8(255 * hmap)#.transpose((1,2,0))
    #hmap_img = cv2.applyColorMap(cv2.resize(hmap_255.transpose((1,2,0)), (width, height)), cv2.COLORMAP_JET)
    hmap_img = cv2.applyColorMap(cv2.resize(hmap_255,(width, height)),cv2.COLORMAP_JET)
    fused = hmap_img * fuse_portions[0] + img * fuse_portions[1]
    fused_img_path = save_file_prefix+'fused_img.jpg'
    #fused_img_path = '/dccstor/jsdata1/dev/2019_projects/StarNet/experiments/LK_starNet_imagenet_loc_1_stage_for_real/dets_1/hm2_0_0fused_img.jpg'
    cv2.imwrite(fused_img_path, fused)
    CAM_output_path = save_file_prefix+'CAM_output.txt'
    curParaThreshold = str(bbox_threshold[0]) + ' ' + str(bbox_threshold[1]) + ' ' + str(bbox_threshold[2]) + ' '
    os.system("CAM_Python_master/bboxgenerator/./dt_box " + fused_img_path + ' ' +
              curParaThreshold + ' ' + CAM_output_path)
    # bboxes ---------------------
    with open(CAM_output_path) as f:
        for line in f:
            items = [int(x) for x in line.strip().split()]
    boxData1 = np.array(items[0::4]).T
    boxData2 = np.array(items[1::4]).T
    boxData3 = np.array(items[2::4]).T
    boxData4 = np.array(items[3::4]).T
    bboxes = np.array([boxData1, boxData2, boxData1+boxData3, boxData2+boxData4]).T
    os.remove(fused_img_path)
    os.remove(CAM_output_path)
    if False:
        # display the box
        disp_img_path = save_file_prefix+'CAM_disp_img.jpg'
        gen_bbox_img(fused_img_path,CAM_output_path, disp_img_path)
    return bboxes
