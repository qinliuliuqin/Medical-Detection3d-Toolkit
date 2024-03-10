# Medical-Detection3d-Toolkit
PyTorch implementation of the volumetric landmark detection engine proposed in the paper [SkullEngine: A Multi-stage CNN Framework for Collaborative CBCT Image Segmentation and Landmark Detection](https://arxiv.org/abs/2110.03828), MICCAI workshop 2021.

<p align="center">
  <img src="./assets/framework.png" alt="drawing", width="700"/>
</p>

## Installation
The code is tested with ``python=3.8.8``, ``torch=1.9.0``, and ``torchvision=0.15.0`` on an A6000 GPU.
```
git clone 
cd Medical-Detection3d-Toolkit
```
Create a new conda environment and install required packages accordingly.
```
conda create -n det3d python=3.8.8
conda activate det3d
pip3 install -r requirements.txt
```

## Data
First, the users need to prepare medical images and their corresponsing landmark annotations. The ``assets`` folder contains an example pair of image (``case_001.nii.gz``) and landmark annoation file (``case_001.csv``). Then, generate landmark masks (e.g, ``case_001_landmark_mask.nii.gz``) given the pairs as demonstrated in this [notebook](./detection3d/scripts/gen_lamdmark_mask.ipynb). The meaning of labels in the landmark mask:
|Label| Meaning|
|----|----|
|``positive integer``| these voxels are ``positive`` samples of landmarks (e.g., voxels with value ``1`` represent the first landmark.)|
|``0``| these voxels are ``negative`` samples (i.e, background voxels)
|``-1``| these are ``boundary`` voxels (i.e., between the positive and negative ones) that are not involved in training. 

Finally, prepare dataset spliting files for training (``train.csv``) and testing (``test.csv``).

## Training
Run the following code for training with a single GPU.
The user may need to modify training settings in ``./config/lmk_train_config.py``. By defult, the model will be saved in ``./saves/weights``.
```
cd detection3d
python lmk_det_train.py --input ./config/lmk_train_config.py --gpus 0
```

    parser.add_argument('-i', '--input', default='../assets/case_001.nii.gz',
                        help='input folder/file for intensity images')
    parser.add_argument('-m', '--model', default='./saves/weights',
                        help='model root folder')
    parser.add_argument('-o', '--output', default='./saves/results',
                        help='output folder for segmentation')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='the gpu id to run model, set to -1 if using cpu only.')


## Evaluation
Run the following code to evaluate a trained model on a single GPU.
```
python lmk_det_infer.py -i ../assets/case_001.nii.gz -m ./saves/weights -o ./saves/results
``` 

## Citation
```bibtex
@article{liu2021skullengine,
  title={SkullEngine: A Multi-stage CNN Framework for Collaborative CBCT Image Segmentation and Landmark Detection},
  author={Liu, Qin and Deng, Han and Lian, Chunfeng and Chen, Xiaoyang and Xiao, Deqiang and Ma, Lei and Chen, Xu and Kuang, Tianshu and Gateno, Jaime and Yap, Pew-Thian and others},
  journal={arXiv preprint arXiv:2110.03828},
  year={2021}
}
```