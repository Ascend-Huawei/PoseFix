import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import math
from dataset import Dataset
from utils import generate_batch, oks_nms
from tqdm import tqdm
os.chdir("/home/HwHiAiUser/leo/samples/python/contrib/posefix/src")
sys.path.append("./common/")

from atlas_utils.acl_resource import AclResource 
from atlas_utils.acl_model import Model


import json

def test_net(input_pose, det_range):

    acl_resource = AclResource()
    acl_resource.init()
        
    model = Model( "../model/posefix.om",)
    dump_results = []
    img_start = det_range[0]
    img_id = 0
    img_id2 = 0
    pbar = tqdm(total=det_range[1] - img_start - 1, position=0)
    pbar.set_description("GPU %s" % str(0))
    while img_start < det_range[1]:
        img_end = img_start + 1
        im_info = input_pose[img_start]
        while img_end < det_range[1] and input_pose[img_end]['image_id'] == im_info['image_id']:
            img_end += 1
        
        # all human detection results of a certain image
        cropped_data = input_pose[img_start:img_end]
        #pbar.set_description("GPU %s" % str(gpu_id))
        pbar.update(img_end - img_start)

        img_start = img_end

        kps_result = np.zeros((len(cropped_data), cfg.num_kps, 3))
        area_save = np.zeros(len(cropped_data))

        # cluster human detection results with test_batch_size
        for batch_id in range(0, len(cropped_data), cfg.test_batch_size):
            start_id = batch_id
            end_id = min(len(cropped_data), batch_id + cfg.test_batch_size)
             
            imgs = []
            input_pose_coords = []
            input_pose_valids = []
            input_pose_scores = []
            crop_infos = []
            for i in range(start_id, start_id + cfg.test_batch_size):
                if i >= end_id:
                    ni = start_id # mannualy padding 
                else:
                    ni = i
                #cropped_data[ni]= {"image_id": 581357, "category_id": 1, "imgpath": "../data/000000581357.jpg", "estimated_joints": [339.134, 166.685, 1.0, 347.953, 160.805, 1.0, 333.254, 160.805, 1.0, 356.772, 151.986, 1.0, 321.496, 151.986, 1.0, 371.471, 163.745, 1.0, 306.797, 172.564, 1.0, 409.687, 140.227, 1.0, 277.4, 190.202, 1.0, 453.782, 110.83, 1.0, 274.46, 201.961, 1.0, 342.074, 246.057, 1.0, 309.737, 248.996, 1.0, 365.591, 222.539, 1.0, 283.28, 240.177, 1.0, 336.194, 293.092, 1.0, 286.219, 319.549, 1.0], "estimated_score": 0.8625, "bbox": [256.52779999999996, 89.95809999999999, 215.18640000000005, 250.46279999999996]}
                img, input_pose_coord, input_pose_valid, input_pose_score, crop_info = generate_batch(cropped_data[ni])
                imgs.append(img)
                input_pose_coords.append(input_pose_coord)
                input_pose_valids.append(input_pose_valid)
                input_pose_scores.append(input_pose_score)
                crop_infos.append(crop_info)
            imgs = np.array(imgs).astype(np.float32)
            input_pose_coords = np.array(input_pose_coords).astype(np.float32)
            input_pose_valids = np.array(input_pose_valids).astype(np.float32)
            input_pose_scores = np.array(input_pose_scores).astype(np.float32)
            crop_infos = np.array(crop_infos)
            # forward
            coord = model.execute([imgs, input_pose_coords,input_pose_valids])[0]
            coord = coord * cfg.input_shape[0] / cfg.output_shape[0]

            if cfg.flip_test:
                flip_imgs = imgs[:, :, ::-1, :].copy()
                flip_input_pose_coords = input_pose_coords.copy()
                flip_input_pose_coords[:,:,0] = cfg.input_shape[1] - 1 - flip_input_pose_coords[:,:,0]
                flip_input_pose_valids = input_pose_valids.copy()
                for (q, w) in cfg.kps_symmetry:
                    flip_input_pose_coords_w, flip_input_pose_coords_q = flip_input_pose_coords[:,w,:].copy(), flip_input_pose_coords[:,q,:].copy()
                    flip_input_pose_coords[:,q,:], flip_input_pose_coords[:,w,:] = flip_input_pose_coords_w, flip_input_pose_coords_q
                    flip_input_pose_valids_w, flip_input_pose_valids_q = flip_input_pose_valids[:,w].copy(), flip_input_pose_valids[:,q].copy()
                    flip_input_pose_valids[:,q], flip_input_pose_valids[:,w] = flip_input_pose_valids_w, flip_input_pose_valids_q
                flip_coord = model.execute([flip_imgs, flip_input_pose_coords, flip_input_pose_valids])

                if flip_coord is not None:
                    flip_coord = flip_coord[0]
                    flip_coord = flip_coord * cfg.input_shape[0] / cfg.output_shape[0]
                else:
                    print("model exec error")

                flip_coord[:,:,0] = cfg.input_shape[1] - 1 - flip_coord[:,:,0]
                for (q, w) in cfg.kps_symmetry:
                    flip_coord_w, flip_coord_q = flip_coord[:,w,:].copy(), flip_coord[:,q,:].copy()
                    flip_coord[:,q,:], flip_coord[:,w,:] = flip_coord_w, flip_coord_q
                coord += flip_coord
                coord /= 2
            coord = coord[0:end_id - start_id]
         
            # for each human detection from clustered batch
            for image_id in range(start_id, end_id):
               
                kps_result[image_id, :, :2] = coord[image_id - start_id]
                kps_result[image_id, :, 2] = input_pose_scores[image_id - start_id]

                vis=False
                crop_info = crop_infos[image_id - start_id,:]
                area = (crop_info[2] - crop_info[0]) * (crop_info[3] - crop_info[1])
                if vis and np.any(kps_result[image_id,:,2]) > 0.9 and area > 96**2:
                    tmpimg = imgs[image_id-start_id].copy()
                    tmpimg = cfg.denormalize_input(tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3,cfg.num_kps))
                    tmpkps[:2,:] = kps_result[image_id,:,:2].transpose(1,0)
                    tmpkps[2,:] = kps_result[image_id,:,2]
                    _tmpimg = tmpimg.copy()
                    _tmpimg = cfg.vis_keypoints(_tmpimg, tmpkps)
                    cv2.imwrite(osp.join(cfg.vis_dir, str(img_id) + '_output.jpg'), _tmpimg)
                    img_id += 1

                # map back to original images
                for j in range(cfg.num_kps):
                    kps_result[image_id, j, 0] = kps_result[image_id, j, 0] / cfg.input_shape[1] * (\
                    crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + crop_infos[image_id - start_id][0]
                    kps_result[image_id, j, 1] = kps_result[image_id, j, 1] / cfg.input_shape[0] * (\
                    crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + crop_infos[image_id - start_id][1]
                
                area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])
                
        #vis
        vis = False
        if vis and np.any(kps_result[:,:,2] > 0.9):
            tmpimg = cv2.imread(os.path.join(cfg.img_path, cropped_data[0]['imgpath']))
            tmpimg = tmpimg.astype('uint8')
            for i in range(len(kps_result)):
                tmpkps = np.zeros((3,cfg.num_kps))
                tmpkps[:2,:] = kps_result[i, :, :2].transpose(1,0)
                tmpkps[2,:] = kps_result[i, :, 2]
                tmpimg = cfg.vis_keypoints(tmpimg, tmpkps)
            cv2.imwrite(osp.join(cfg.vis_dir, str(img_id2) + '.jpg'), tmpimg)
            img_id2 += 1
        
        # oks nms
        if cfg.dataset in ['COCO', 'PoseTrack']:
            nms_kps = np.delete(kps_result,cfg.ignore_kps,1)
            nms_score = np.mean(nms_kps[:,:,2],axis=1)
            nms_kps[:,:,2] = 1
            nms_kps = nms_kps.reshape(len(kps_result),-1)
            nms_sigmas = np.delete(cfg.kps_sigmas,cfg.ignore_kps)
            keep = oks_nms(nms_kps, nms_score, area_save, cfg.oks_nms_thr, nms_sigmas)
            if len(keep) > 0 :
                kps_result = kps_result[keep,:,:]
                area_save = area_save[keep]
 
        score_result = np.copy(kps_result[:, :, 2])
        kps_result[:, :, 2] = 1
        kps_result = kps_result.reshape(-1,cfg.num_kps*3)
       
        # save result
        for i in range(len(kps_result)):
            if cfg.dataset == 'COCO':
                result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(np.mean(score_result[i]), 4)),
                             keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'PoseTrack':
                result = dict(image_id=im_info['image_id'], category_id=1, track_id=0, scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'MPII':
                result = dict(image_id=im_info['image_id'], scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())

            dump_results.append(result)

    return dump_results


def test():
    
    # annotation load
    d = Dataset()
    annot = d.load_annot("val")

    # input pose load
    #input_pose = d.input_pose_load(annot, "val")
    
    
   
    
    with open("../data/input_pose.json","r") as f:
        input_pose = json.load(f)
 
    img_start = 0
    ranges = [0]
    img_num = len(np.unique([i['image_id'] for i in input_pose]))
    images_per_gpu = img_num + + 1
    for run_img in range(img_num):
        img_end = img_start + 1
        while img_end < len(input_pose) and input_pose[img_end]['image_id'] == input_pose[img_start]['image_id']:
            img_end += 1
        if (run_img + 1) % images_per_gpu == 0 or (run_img + 1) == img_num:
            ranges.append(img_end)
        img_start = img_end

    result = test_net(input_pose, ranges)

    
    # evaluation
    d.evaluation(result, annot, cfg.result_dir, cfg.testset)

if __name__ == '__main__':
    
    test()
