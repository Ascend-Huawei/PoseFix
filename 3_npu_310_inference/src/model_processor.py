# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions andprint
limitations under the License.
"""

import os
import cv2
import numpy as np
import sys

sys.path.append("./common/")

from atlas_utils.acl_model import Model
from utils import generate_batch, normalize_input, denormalize_input, vis_keypoints

heatmap_width = 92
heatmap_height = 92


class ModelProcessor(object):
    """acl model wrapper"""
    def __init__(self, acl_resource, params):
        self._acl_resource = acl_resource
        self.params = params

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = Model(params['model_dir'])
        self.input_shape = params["input_shape"]
        self.num_kps = params["num_kps"]
        self.output_shape = params["output_shape"]

    def predict(self, d):
        """run predict"""
        #preprocess image to get 'model_input'
        img, input_pose_coord, input_pose_valid, input_pose_score, crop_info = self.preprocess(d)

        # execute model inference
        result = self.model.execute([img, input_pose_coord,input_pose_valid])
        # postprocessing: use the heatmaps (the output of model) to get the joins and limbs for human body
        # Note: the model has multiple outputs, here we used a simplified method, which only uses heatmap for body joints
        #       and the heatmap has shape of [1,14], each value correspond to the position of one of the 14 joints. 
        #       The value is the index in the 92*92 heatmap (flatten to one dimension)
        coord = result[0][0]
    
        coord = coord  *self.input_shape[0] / self.output_shape[0]
        # calculate the scale of original image over heatmap, Note: image_original.shape[0] is height
        kps_result = np.zeros((17, 3))
        kps_result[:, :2] = coord
        kps_result[:, 2] = input_pose_score

        #cropped keypoints image
        tmpimg = img[0].copy()
        tmpimg = denormalize_input(tmpimg)
        tmpimg = tmpimg.astype('uint8')
        tmpkps = np.zeros((3,self.num_kps))
        tmpkps[:2,:] = kps_result[:,:2].transpose(1,0)
        tmpkps[2,:] = kps_result[:,2]
        _tmpimg = tmpimg.copy()
        _tmpimg = vis_keypoints(_tmpimg, tmpkps)
        

        # full image keypoints
        for j in range(self.num_kps):
            kps_result[j, 0] = kps_result[j, 0] / self.input_shape[1] * (crop_info[2] - crop_info[0]) + crop_info[0]
            kps_result[j, 1] = kps_result[j, 1] / self.input_shape[0] * (crop_info[3] - crop_info[1]) + crop_info[1]
                
        tmpimg = cv2.imread(d['imgpath'])
        tmpimg = tmpimg.astype('uint8')
       
        tmpkps = np.zeros((3,self.num_kps))
        tmpkps[:2,:] = kps_result[ :, :2].transpose(1,0)
        tmpkps[2,:] = kps_result[:, 2]
        tmpimg = vis_keypoints(tmpimg, tmpkps)

        return _tmpimg, tmpimg

    def preprocess(self, d):
        """
        preprocessing: resize image to model required size, and normalize value between [0,1]
        """
        img, input_pose_coord, input_pose_valid, input_pose_score, crop_info = generate_batch(d)

        img = np.tile(img[np.newaxis,:],[16,1,1,1]).astype(np.float32)
        input_pose_coord = np.tile(input_pose_coord[np.newaxis,:],[16,1,1]).astype(np.float32)
        input_pose_valid = np.tile(input_pose_valid[np.newaxis,:],[16,1]).astype(np.float32) 
        
        return img, input_pose_coord, input_pose_valid, input_pose_score, crop_info
        
 

