import glob
import importlib
import numpy as np
import torch.nn as nn
import os
import pandas as pd
import SimpleITK as sitk
import time
import torch
from easydict import EasyDict as edict

from detection3d.utils.file_io import load_config
from detection3d.utils.model_io import get_checkpoint_folder
from detection3d.utils.image_tools import convert_image_to_tensor, convert_tensor_to_image, \
    resample_spacing, pick_largest_connected_component, weighted_voxel_center
from detection3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer
from detection3d.dataset.landmark_dataset import read_image_list


def read_test_folder(folder_path):
    """ read single-modality input folder
    :param folder_path: image file folder path
    :return: a list of image path list, list of image case names
    """
    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    file = []
    for suf in suffix:
        file += glob.glob(os.path.join(folder_path, '*' + suf))

    file_name_list, file_path_list = [], []
    for im_pth in sorted(file):
        _, im_name = os.path.split(im_pth)
        for suf in suffix:
            idx = im_name.find(suf)
            if idx != -1:
                im_name = im_name[:idx]
                break
        file_name_list.append(im_name)
        file_path_list.append(im_pth)

    return file_name_list, file_path_list


def load_det_model(model_folder, gpu_id=0):
    """ load segmentation model from folder
    :param model_folder:    the folder containing the segmentation model
    :param gpu_id:          the gpu device id to run the segmentation model
    :return: a dictionary containing the model and inference parameters
    """
    assert os.path.isdir(model_folder), 'Model folder does not exist: {}'.format(
        model_folder)

    # load inference config file
    latest_checkpoint_dir = get_checkpoint_folder(
        os.path.join(model_folder, 'checkpoints'), -1)
    infer_cfg = load_config(
        os.path.join(latest_checkpoint_dir, 'lmk_infer_config.py'))

    model = edict()
    model.infer_cfg = infer_cfg
    train_cfg = load_config(
        os.path.join(latest_checkpoint_dir, 'lmk_train_config.py'))
    model.train_cfg = train_cfg

    # load model state
    chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')

    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(int(gpu_id))
        # load network module
        state = torch.load(chk_file)
        net_module = importlib.import_module(
            'detection.network.' + state['net'])
        net = net_module.Net(state['in_channels'], state['num_landmark_classes'] + 1)
        net = nn.parallel.DataParallel(net, device_ids=[0])
        net.load_state_dict(state['state_dict'])
        net.eval()
        net = net.cuda()
        del os.environ['CUDA_VISIBLE_DEVICES']

    else:
        state = torch.load(chk_file, map_location='cpu')
        net_module = importlib.import_module(
            'detection.network.' + state['net'])
        net = net_module.Net(state['in_channels'], state['num_landmark_classes'] + 1)
        net.load_state_dict(state['state_dict'])
        net.eval()

    model.net = net
    model.crop_size, model.crop_spacing, model.max_stride, model.interpolation = \
        state['crop_size'], state['crop_spacing'], state['max_stride'], state['interpolation']
    model.in_channels, model.num_landmark_classes = \
        state['in_channels'], state['num_landmark_classes']

    model.crop_normalizers = []
    for crop_normalizer in state['crop_normalizers']:
        if crop_normalizer['type'] == 0:
            mean, stddev, clip = crop_normalizer['mean'], crop_normalizer['stddev'], \
                                 crop_normalizer['clip']
            model.crop_normalizers.append(FixedNormalizer(mean, stddev, clip))

        elif crop_normalizer['type'] == 1:
            clip_sigma = crop_normalizer['clip_sigma']
            model.crop_normalizers.append(AdaptiveNormalizer(clip_sigma))

        else:
            raise ValueError('Unsupported normalization type.')

    return model


def detection_voi(model, iso_image, start_voxel, end_voxel, use_gpu):
    """ Segment the volume of interest
    :param model:           the loaded segmentation model.
    :param iso_image:       the image volume that has the same spacing with the model's resampling spacing.
    :param start_voxel:     the start voxel of the volume of interest (inclusive).
    :param end_voxel:       the end voxel of the volume of interest (exclusive).
    :param use_gpu:         whether to use gpu or not, bool type.
    :return:
      mean_prob_maps:        the mean probability maps of all classes
      std_maps:              the standard deviation maps of all classes
    """
    assert isinstance(iso_image, sitk.Image)

    roi_image = iso_image[start_voxel[0]:end_voxel[0],
                start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2]]

    if model['crop_normalizers'] is not None:
        roi_image = model.crop_normalizers[0](roi_image)

    roi_image_tensor = convert_image_to_tensor(roi_image).unsqueeze(0)
    if use_gpu:
        roi_image_tensor = roi_image_tensor.cuda()

    with torch.no_grad():
        landmarks_pred = model['net'](roi_image_tensor)

    return landmarks_pred


def detection_single_image(image, image_name, model, gpu_id, save_prob, save_folder):
    """ volumetric segmentation for single image
    :param image: the input volume
    :param image_name: the name of the image
    :param model: the detection model
    :param gpu_id: the id of the gpu
    :return: a dictionary containing the detected landmarks
    """
    assert isinstance(image, sitk.Image)

    # load landmark label dictionary
    landmark_dict = model['train_cfg'].general.target_landmark_label
    landmark_name_list, landmark_label_list = [], []
    for landmark_name in landmark_dict.keys():
        landmark_name_list.append(landmark_name)
        landmark_label_list.append(landmark_dict[landmark_name])
    landmark_label_reorder = np.argsort(landmark_label_list)

    iso_image = resample_spacing(image, model['crop_spacing'], model['max_stride'],
                                 model['interpolation'])
    assert isinstance(iso_image, sitk.Image)

    start_voxel, end_voxel = [0, 0, 0], [int(iso_image.GetSize()[idx]) for idx in range(3)]
    voi_landmarks_pred = detection_voi(model, iso_image, start_voxel, end_voxel, gpu_id > 0)
    print('{:0.2f}%'.format( 100))

    # convert to landmark masks
    landmark_mask_preds = voi_landmarks_pred.cpu()
    assert landmark_mask_preds.shape[0] == 1
    landmark_mask_preds = torch.squeeze(landmark_mask_preds)
    landmark_mask_preds = convert_tensor_to_image(landmark_mask_preds, sitk.sitkFloat32)

    detected_landmark = []
    for i in range(0, model['num_landmark_classes']):
        landmark_mask_pred = landmark_mask_preds[i + 1]  # exclude the background
        landmark_mask_pred.CopyInformation(iso_image)

        landmark_mask_prob = sitk.GetArrayFromImage(landmark_mask_pred)
        # threshold the probability map to get the binary mask
        prob_threshold = 0.5
        landmark_mask_binary = np.zeros_like(landmark_mask_prob, dtype=np.int16)
        landmark_mask_binary[landmark_mask_prob >= prob_threshold] = 1
        landmark_mask_binary[landmark_mask_prob < prob_threshold] = 0

        # pick the largest connected component
        landmark_mask_cc = sitk.GetImageFromArray(landmark_mask_binary)
        landmark_mask_cc = pick_largest_connected_component(landmark_mask_cc, [1])

        # only keep probability of the largest connected component
        landmark_mask_cc = sitk.GetArrayFromImage(landmark_mask_cc)
        masked_landmark_mask_prob = np.multiply(landmark_mask_cc.astype(np.float), landmark_mask_prob)

        # compute the weighted mass center of the probability map
        masked_landmark_mask_prob = sitk.GetImageFromArray(masked_landmark_mask_prob)
        masked_landmark_mask_prob.CopyInformation(iso_image)
        voxel_coordinate = weighted_voxel_center(masked_landmark_mask_prob, prob_threshold, 1.0)

        landmark_name = landmark_name_list[landmark_label_reorder[i]]
        if voxel_coordinate is not None:
            world_coordinate = masked_landmark_mask_prob.TransformContinuousIndexToPhysicalPoint(voxel_coordinate)
            print("world coordinate of volume {0} landmark {1} is:[{2},{3},{4}]".format(
                image_name, i, world_coordinate[0], world_coordinate[1], world_coordinate[2]))
            detected_landmark.append(
                [landmark_name, world_coordinate[0], world_coordinate[1], world_coordinate[2]]
            )
        else:
            print("world coordinate of volume {0} landmark {1} is not detected.".format(image_name, i))
            detected_landmark.append([landmark_name, -1, -1, -1])

    detected_landmark_df = pd.DataFrame(data=detected_landmark, columns=['name', 'x', 'y', 'z'])

    return detected_landmark_df


def detection(input_path, model_folder, gpu_id, return_landmark_file, save_landmark_file, save_prob, output_folder):
    """ volumetric image segmentation engine
    :param input_path:              The path of text file, a single image file or a root dir with all image files
    :param model_folder:            The path of trained model
    :param gpu_id:                  Which gpu to use, set as -1 to use CPU for inference
    :param return_landmark_file:    Whether return landmark file
    :param save_landmark_file:      Whether save landmark file to disk
    :param save_prob:               Whether save probability maps for all detected landmarks
    :param output_folder:           The path of out folder
    :return: None
    """

    # load model
    begin = time.time()
    model = load_det_model(model_folder, gpu_id)
    load_model_time = time.time() - begin

    # load landmark label dictionary
    landmark_dict = model['train_cfg'].general.target_landmark_label
    landmark_name_list, landmark_label_list = [], []
    for landmark_name in landmark_dict.keys():
        landmark_name_list.append(landmark_name)
        landmark_label_list.append(landmark_dict[landmark_name])
    landmark_label_reorder = np.argsort(landmark_label_list)

    # load test images
    if os.path.isfile(input_path):
        if input_path.endswith('.csv'):
            file_name_list, file_path_list = read_image_list(input_path, 'test')
        else:
            if input_path.endswith('.mhd') or input_path.endswith('.mha') or \
                    input_path.endswith('.nii.gz') or input_path.endswith('.nii') or \
                    input_path.endswith('.hdr') or input_path.endswith('.image3d'):
                im_name = os.path.basename(input_path)
                file_name_list = [im_name]
                file_path_list = [input_path]

            else:
                raise ValueError('Unsupported input path.')

    elif os.path.isdir(input_path):
        file_name_list, file_path_list = read_test_folder(input_path)

    else:
        if input_path.endswith('.csv'):
            raise ValueError('The file doest no exist: {}.'.format(input_path))
        else:
            raise ValueError('Unsupported input path.')

    if save_landmark_file or save_prob:
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

    # test each case
    landmark_files = []
    for i, file_path in enumerate(file_path_list):
        print('{}: {}'.format(i, file_path))

        if not os.path.isfile(file_path):
            print('File {} does not exist!'.format(file_path))
            continue

        # load image
        begin = time.time()
        image = sitk.ReadImage(file_path, sitk.sitkFloat32)
        read_image_time = time.time() - begin

        iso_image = resample_spacing(image, model['crop_spacing'], model['max_stride'],
                                     model['interpolation'])
        assert isinstance(iso_image, sitk.Image)

        start_voxels = [[0, 0, 0]]
        end_voxels = [[int(iso_image.GetSize()[idx]) for idx in range(3)]]

        begin = time.time()
        voi_landmark_mask_preds = []
        for idx in range(len(start_voxels)):
            start_voxel, end_voxel = start_voxels[idx], end_voxels[idx]

            voi_landmarks_pred = \
                detection_voi(model, iso_image, start_voxel, end_voxel, gpu_id > 0)

            voi_landmark_mask_preds.append(voi_landmarks_pred)
            print('{:0.2f}%'.format((idx + 1) / len(start_voxels) * 100))

        # convert to landmark masks
        landmark_mask_preds = voi_landmark_mask_preds[0].cpu()
        assert landmark_mask_preds.shape[0] == 1
        landmark_mask_preds = torch.squeeze(landmark_mask_preds)
        landmark_mask_preds = convert_tensor_to_image(landmark_mask_preds, sitk.sitkFloat32)
        inference_time = time.time() - begin

        begin = time.time()
        detected_landmark = []
        for j in range(0, model['num_landmark_classes']):
          landmark_mask_pred = landmark_mask_preds[j + 1] # exclude the background
          landmark_mask_pred.CopyInformation(iso_image)

          if save_prob:
            prob_path = os.path.join(output_folder, '{}_{}.mha'.format(file_name_list[i], j))
            sitk.WriteImage(landmark_mask_pred, prob_path)

          landmark_mask_prob = sitk.GetArrayFromImage(landmark_mask_pred)
          # threshold the probability map to get the binary mask
          prob_threshold = 0.5
          landmark_mask_binary = np.zeros_like(landmark_mask_prob, dtype=np.int16)
          landmark_mask_binary[landmark_mask_prob >= prob_threshold] = 1
          landmark_mask_binary[landmark_mask_prob < prob_threshold] = 0

          # pick the largest connected component
          landmark_mask_cc = sitk.GetImageFromArray(landmark_mask_binary)
          landmark_mask_cc = pick_largest_connected_component(landmark_mask_cc, [1])

          # only keep probability of the largest connected component
          landmark_mask_cc = sitk.GetArrayFromImage(landmark_mask_cc)
          masked_landmark_mask_prob = np.multiply(landmark_mask_cc.astype(np.float), landmark_mask_prob)

          # compute the weighted mass center of the probability map
          masked_landmark_mask_prob = sitk.GetImageFromArray(masked_landmark_mask_prob)
          masked_landmark_mask_prob.CopyInformation(iso_image)
          voxel_coordinate = weighted_voxel_center(masked_landmark_mask_prob, prob_threshold, 1.0)

          landmark_name = landmark_name_list[landmark_label_reorder[j]]
          if voxel_coordinate is not None:
            world_coordinate = masked_landmark_mask_prob.TransformContinuousIndexToPhysicalPoint(voxel_coordinate)
            print("world coordinate of volume {0} landmark {1} is:[{2},{3},{4}]".format(
              file_name_list[i], j, world_coordinate[0], world_coordinate[1], world_coordinate[2]))
            detected_landmark.append(
                [landmark_name, world_coordinate[0], world_coordinate[1], world_coordinate[2]]
            )
          else:
            print("world coordinate of volume {0} landmark {1} is not detected.".format(file_name_list[i], j))
            detected_landmark.append([landmark_name, -1, -1, -1])

        detected_landmark_df = pd.DataFrame(data=detected_landmark, columns=['name', 'x', 'y', 'z'])
        if return_landmark_file:
            landmark_files.append(detected_landmark_df)

        if save_landmark_file:
            detected_landmark_save_path = os.path.join(output_folder, '{}.csv'.format(file_name_list[i]))
            detected_landmark_df.to_csv(detected_landmark_save_path, index=False)

        saving_time = time.time() - begin
        print('read: {:.2f} s, prediction: {:.2f} s, saving: {:.2f} s'.format(
            read_image_time + load_model_time, inference_time, saving_time)
        )

    return landmark_files