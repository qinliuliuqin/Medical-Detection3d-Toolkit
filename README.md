# Medical-Detection3d-Toolkit
Landmark detection engine for 3D medical images

# Framework
![avatar](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/framework.png)

# How to use?
## Training
1. Clone the code repository.
```
$ git clone https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit.git
```
2. Generate the landmark mask using this [code](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/detection3d/scripts/gen_landmark_mask.py).
An example landmark mask can be found [here](https://github.com/qinliuliuqin/Model-Zoo/blob/master/Dental/detection/landmark/test_data/landmark_mask.mha). 

In the landmark mask, there are three types of voxels:
|Voxel value| Voxel type|
|----|----|
|Positive: 1 - N| landmark voxels-voxels of different landmarks should have different voxel value (e.g., for the landmark ith, voxels of this landmark can be set to i.)|
|0| background voxels
|-1| invalid voxles  

3. Generate the training file, [here](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/demo/train.csv) is an example.

4. Train the model using this [code](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/detection3d/lmk_det_train.py). [Here](https://github.com/qinliuliuqin/Model-Zoo/blob/master/Dental/detection/landmark/model_0531_2020/batch_1/checkpoints/chk_1200/lmk_train_config.py) is an example configuration file.

# Inference
