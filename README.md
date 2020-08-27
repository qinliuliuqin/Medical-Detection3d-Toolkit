# Medical-Detection3d-Toolkit
Landmark detection engine for 3D medical images

# Framework
![avatar](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/framework.png)

We use [V-Net](http://far.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf)-a variant of 3D U-Net-for landmark detection. The overall landmark detection pipeline is very similar to the segmentation pipeline. 

## Training
We first generate the landmark mask for each intensity image. You may ask that each landmark is a world coordinate, how can we generate a mask for a coordinate? Good question! We solve it by taking all the voxels within a predefined physical distance (e.g., 3mm) of a landmark's coordinate as the voxels of that landmark. In this way, we can generate a multi-class mask as shown in the above picture. In the landmark mask, the value of landmark voxels are set to positive, the value of background voxels are set to zero. To make our model more robust, we set value of the background voxels that are very close to the landmark voxels (e.g., < 3mm) to negative, so they are regarded as invalid voxels that won't be computed in the loss function. 

We then train a multi-class segmentation model given the generated landmark mask for each original image. The detals of training a segmentation can be referred to the [V-Net](http://far.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf) paper. The only difference is that we use [Focal loss](https://arxiv.org/abs/1708.02002)-instead of Dice loss that was proposed in the paper-as the loss function. We found Focal loss was more stable than the Dice loss.

## Inference
For a given test image and the trained segmentation model, we first obtain the probability maps for all landmarks. As shown in the above picture, each probability map represents the prediction of the landmark. We can simply pick the voxel with the highest probability as the final prediction. Or we can use the weighed voxel center as the final prediction. The latter method is more accurate and robust than the former one. 


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

3. Generate the training file, an example file can be found [here](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/demo/train.csv).

4. Train the model using this [code](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/detection3d/lmk_det_train.py), an example configuration file can be found [Here](https://github.com/qinliuliuqin/Model-Zoo/blob/master/Dental/detection/landmark/model_0531_2020/batch_1/checkpoints/chk_1200/lmk_train_config.py).

# Inference
[lmk_det_infer.py]()
