
# GPU doc

## Training Dataset
COCO dataset
```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
```
Extra files 
person_keypoints_256x192_resnet50_val2017_results.json

https://drive.google.com/drive/folders/10d8oSlCnWD-n3CBARXFj7kDBOKXfSYmf 

result.json 

https://drive.google.com/drive/folders/1blrmQ2CRiCeJDjx7sFzLFOBtlALf2Uxa  

Model files
```
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
```


## Repo https://github.com/mks0601/PoseFix_RELEASE
Clone
```
git clone https://github.com/mks0601/PoseFix_RELEASE
```
Follow the original repo guide to put all the data as the following structure. The person_keypoints_256x192_resnet50_val2017_results should replace the name_of_input_pose.json under ${POSE_ROOT}data/COCO/input_pose.

```
${POSE_ROOT}
|-- data
|-- |-- MPII
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- train.json
|       |   `-- test.json
|       `-- images
|           |-- 000001163.jpg
|           |-- 000003072.jpg
|-- |-- PoseTrack
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- train2018.json
|       |   |-- val2018.json
|       |   `-- test2018.json
|       |-- original_annotations
|       |   |-- train/
|       |   |-- val/
|       |   `-- test/
|       `-- images
|           |-- train/
|           |-- val/
|           `-- test/
|-- |-- COCO
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- person_keypoints_train2017.json
|       |   |-- person_keypoints_val2017.json
|       |   `-- image_info_test-dev2017.json
|       `-- images
|           |-- train2017/
|           |-- val2017/
|           `-- test2017/
`-- |-- imagenet_weights
|       |-- resnet_v1_50.ckpt
|       |-- resnet_v1_101.ckpt
|       `-- resnet_v1_152.ckpt
```

## GPU training command

```
cd main
python train.py --gpu 0
```
OR
```
bash run_1p.sh
```

## pbtxt and ckpt and training_log
checkpoint and pbtxt in https://drive.google.com/drive/folders/15ANznISxzazKQDdl2oz6jUqWLcBLodZz?usp=sharing

## evaluation result
| Experiment        | AP | AP(0.5) | AP(0.75) |  APM  | APL | AR | AR(0.5) | AR(0.75) |ARM | ARL |
|:-----------------:|:---------:|:---------:|:----------:|:-------:|:-----------------:|:---------:|:---------:|:----------:|:-----:|-----|
|Original Paper|72.5|90.5|79.6|68.9|79.0|78.0|94.1|84.4|73.4|84.1|
|Trained model |73.2|89.3|79.5|69.7|80.1|78.6|93.2|84.3|74.3|84.9|



