# NPU inference on 200dk board doc
## Convert pb to om model
Get the pb file from https://onebox.huawei.com/p/594e2dda144f45dc8b356722d88c98b and run the atc command below to convert the model
```
 atc --framework=3 --model=posefix.pb --input_format=ND --input_shape="Placeholder:16,384,288,3;Placeholder_2:16,17,2;Placeholder_4:16,17"  --output=posefix --output_type=FP32 --out_nodes="mul_15:0" --soc_version=Ascend310
```

## One image inference
Put json file and image in the data folder.
Get the model from https://onebox.huawei.com/p/b94012a151adc69e42b418a01a00b33a \
and run the following script in the src folder
```
python3 run_image.py --model ../model/posefix.om --input_json ../data/test.json
```
The json file should follow the format of the test.json where the estimated_joints are the key points coordinates predicted and bbox is the person's bounding box in the image. 

## Evaluation on 2017 coco dataset
Get COCO dataset
```
wget http://images.cocodataset.org/zips/val2017.zip
```
Extract the dataset in the data folder and run the following script in the src folder.
```
python3 eval.py
```

## evaluation result
| Experiment        | AP | AP(0.5) | AP(0.75) |  APM  | APL | AR | AR(0.5) | AR(0.75) |ARM | ARL |
|:-----------------:|:---------:|:---------:|:----------:|:-------:|:-----------------:|:---------:|:---------:|:----------:|:-----:|-----|
|Original Paper|72.5|90.5|79.6|68.9|79.0|78.0|94.1|84.4|73.4|84.1|
|Trained model GPU|73.2|89.3|79.5|69.7|80.1|78.6|93.2|84.3|74.3|84.9|
|Trained model NPU|73.3|89.3|79.6|69.8|80.1|78.6|93.3|84.4|74.4|84.9|
|Inference on 200DK|72.8|89.2|79.3|69.2|79.7|78.3|93.2|84.1|73.9|84.5
