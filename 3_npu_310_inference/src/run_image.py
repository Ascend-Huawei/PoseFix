import os
import cv2
import numpy as np
import argparse
import sys
import json
sys.path.append("./common/")

from atlas_utils.acl_resource import AclResource 
from model_processor import ModelProcessor

def main(model, input_json, output_dir):
    d = json.load(open(input_json,"r"))

    acl_resource = AclResource()
    acl_resource.init()
        
    model_parameters = {
                        'model_dir': model,
                        'input_shape': (384, 288),
                        'output_shape': (96, 72),
                        'num_kps':17
                        }
    
    # perpare model instance: init (loading model from file to memory)
    # model_processor: preprocessing + model inference + postprocessing
    model_processor = ModelProcessor(acl_resource, model_parameters)
    cropped_img, full_img = model_processor.predict(d)

    cv2.imwrite(os.path.join(output_dir, str(d["image_id"]) + "_cropped.jpg"),cropped_img)
    cv2.imwrite(os.path.join(output_dir, str(d['image_id']) + "_full.jpg"), full_img)

if __name__=='__main__':
    description = 'Load a model for posefix'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default="../model/posefix.om")
    parser.add_argument('--input_json', type=str, default="../data/test.json", help="json for the input")
    parser.add_argument('--output_dir', type=str, default="../output", help="Output Path")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args.model, args.input_json, args.output_dir)
# atc --framework=3 --model=changed_graph_posefix.pb --input_format=ND --input_shape="Placeholder:16,384,288,3;Placeholder_2:16,17,2;Placeholder_4:16,17"  --output=posefix --output_type=FP32 --out_nodes="mul_15:0" --soc_version=Ascend310
# atc --framework=3 --model=five_inputs_model.pb --input_format=ND --input_shape="Placeholder_2:16,17,2;Placeholder_4:16,17"  --output=five_inputs_model --output_type=FP32 --out_nodes="StopGradient:0" --soc_version=Ascend310