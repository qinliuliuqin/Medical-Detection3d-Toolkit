from __future__ import print_function
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torchio as tio

from detection3d.utils.image_tools import select_random_voxels_in_multi_class_mask, \
  crop_image, convert_image_to_tensor, get_image_frame, pad_image
from detection3d.utils.landmark_utils import is_world_coordinate_valid, \
  is_voxel_coordinate_valid


def read_landmark_coords(image_name_list, landmark_file_path, target_landmark_label):
  """
  Read a list of labelled landmark csv files and return a list of labelled
  landmarks.
  """
  assert len(image_name_list) == len(landmark_file_path)

  label_dict = {}
  for idx, image_name in enumerate(image_name_list):
    label_dict[image_name] = {}
    label_dict[image_name]['label'] = []
    label_dict[image_name]['name'] = []
    label_dict[image_name]['coords'] = []
    landmark_file_df = pd.read_csv(landmark_file_path[idx])

    for row_idx in range(len(landmark_file_df)):
      landmark_name = landmark_file_df['name'][row_idx]
      if landmark_name in target_landmark_label.keys():
        landmark_label = target_landmark_label[landmark_name]
        x = landmark_file_df['x'][row_idx]
        y = landmark_file_df['y'][row_idx]
        z = landmark_file_df['z'][row_idx]
        landmark_coords = [x, y, z]
        label_dict[image_name]['label'].append(landmark_label)
        label_dict[image_name]['name'].append(landmark_name)
        label_dict[image_name]['coords'].append(landmark_coords)

    assert len(label_dict[image_name]['name']) == len(target_landmark_label.keys())

  return label_dict


def read_image_list(image_list_file, mode):
    """
    Reads the image list CSV file and returns file paths depending on the mode.

    Expected CSV structure:
    -----------------------------------------------------------
    For 'train' or 'val' mode:
        Columns:
            - image_name           : string identifier for the image
            - image_path           : path to the image file
            - landmark_file_path   : path to landmark file (e.g., JSON/CSV/TXT)
            - landmark_mask_path   : path to landmark mask file (e.g., NIfTI/PNG)

    For 'test' mode:
        Columns:
            - image_name           : string identifier for the image
            - image_path           : path to the image file

    Returns:
        train/val → (image_name_list, image_path_list, landmark_file_path_list, landmark_mask_path_list)
        test      → (image_name_list, image_path_list, None, None)
    """

    images_df = pd.read_csv(image_list_file)

    # Basic required columns
    required_cols = ['image_name', 'image_path']
    for col in required_cols:
        assert col in images_df.columns, f"Missing required column: '{col}' in CSV"

    image_name_list = images_df['image_name'].tolist()
    image_path_list = images_df['image_path'].tolist()

    if mode == 'test':
        return image_name_list, image_path_list, None, None

    elif mode in ['train', 'val']:
        # Additional required columns for train/val
        extra_cols = ['landmark_file_path', 'landmark_mask_path']
        for col in extra_cols:
            assert col in images_df.columns, f"Missing required column: '{col}' for mode='{mode}'"

        landmark_file_path_list = images_df['landmark_file_path'].tolist()
        landmark_mask_path_list = images_df['landmark_mask_path'].tolist()

        return image_name_list, image_path_list, landmark_file_path_list, landmark_mask_path_list

    else:
        raise ValueError(f"Unsupported mode type: {mode}")


class LandmarkDetectionDataset(Dataset):
  """
  Training dataset for multi-landmark detection.
  """
  def __init__(self,
                mode,
                image_list_file,
                target_landmark_label,
                crop_size,
                pad_size,
                crop_spacing,
                sampling_method,
                sampling_size,
                positive_upper_bound,
                negative_lower_bound,
                num_pos_patches_per_image,
                num_neg_patches_per_image,
                augmentation_turn_on,
                aug_cfg,
                interpolation,
                crop_normalizers):
    
    self.mode = mode.lower()
    assert self.mode in ['train', 'val']

    self.image_name_list, self.image_path_list, self.landmark_file_path, self.landmark_mask_path = \
      read_image_list(image_list_file, self.mode)
    assert len(self.image_name_list) == len(self.image_path_list)

    self.target_landmark_label = target_landmark_label
    self.landmark_coords_dict = read_landmark_coords(
      self.image_name_list, self.landmark_file_path, self.target_landmark_label
    )
    self.crop_spacing = np.array(crop_spacing, dtype=np.float32)
    self.crop_size = np.array(crop_size, dtype=np.float32)
    self.pad_size = pad_size
    self.sampling_method = sampling_method
    self.sampling_size = sampling_size
    self.positive_upper_bound = positive_upper_bound
    self.negative_lower_bound = negative_lower_bound
    self.num_pos_patches_per_image = num_pos_patches_per_image
    assert self.num_pos_patches_per_image >= 0
    self.num_neg_patches_per_image = num_neg_patches_per_image
    assert self.num_neg_patches_per_image >= 0
    # + 1 for background
    self.num_landmark_classes = len(target_landmark_label) + 1
    self.augmentation_turn_on = augmentation_turn_on
    self.interpolation = interpolation
    self.crop_normalizers = crop_normalizers

    if self.augmentation_turn_on:
      self.aug_transform = self.build_transform(aug_cfg)

    self.augmentation_translation_lmk = aug_cfg.translation_lmk

    self.positive_perturbs = []
    for dz in range(-positive_upper_bound, positive_upper_bound):
      for dy in range(-positive_upper_bound, positive_upper_bound):
        for dx in range(-positive_upper_bound, positive_upper_bound):
          perturb = [dx, dy, dz]
          if np.linalg.norm(perturb) <= positive_upper_bound:
            self.positive_perturbs.append(perturb)

  def __len__(self):
    """ get the number of images in this dataset """
    return len(self.image_name_list)

  def num_modality(self):
    """ get the number of input image modalities """
    return 1

  def num_landmark_classes(self):
    return self.num_landmark_classes

  def num_organ_classes(self):
    return self.num_organ_classes

  def global_sample(self, image):
      """ random sample a position in the image
      :param image: a SimpleITK image object which should be in the RAI coordinate
      :return: a world position in the RAI coordinate
      """
      assert isinstance(image, sitk.Image)

      origin = image.GetOrigin()
      im_size_mm = [image.GetSize()[idx] * image.GetSpacing()[idx] for idx in range(3)]
      crop_size_mm = self.crop_size * self.crop_spacing

      im_size_mm = np.array(im_size_mm, dtype=np.double)
      crop_size_mm = np.array(crop_size_mm, dtype=np.double)
      origin = np.array(origin, dtype=np.double)

      # Compute random starting point within bounds
      max_offsets_mm = np.clip(im_size_mm - crop_size_mm, 0, None)
      crop_corner_start = origin + np.random.uniform(0, 1, size=3) * max_offsets_mm

      crop_center_mm = crop_corner_start + crop_size_mm / 2
      return crop_center_mm

  def center_sample(self, image):
    """ return the world coordinate of the image center
    :param image: a image3d object
    :return: the image center in world coordinate
    """
    assert isinstance(image, sitk.Image)

    origin = image.GetOrigin()
    end_point_voxel = [int(image.GetSize()[idx] - 1) for idx in range(3)]
    end_point_world = image.TransformIndexToPhysicalPoint(end_point_voxel)

    center = np.array([(origin[idx] + end_point_world[idx]) / 2.0 for idx in range(3)], dtype=np.double)
    return center

  def select_samples_in_the_landmark_mask(self, landmark_mask, landmark_df):
    """
    Select samples from the landmark mask.
    :param landmark_mask: The landmark mask in which voxels of different landmarks have different labels.
    :param landmark_df: The dictionary of the landmark coordinates.
    :return: the landmark mask that has a balanced positive and negative sample voxels.
    """
    assert isinstance(landmark_mask, sitk.Image)

    mask_npy = sitk.GetArrayFromImage(landmark_mask)
    selected_mask_npy = np.zeros_like(mask_npy) - 1

    sample_voxels = []

    # set positive samples, which are centered at the landmark coordinates.
    valid_landmark_idx = []
    for idx in range(len(landmark_df['name'])):
      landmark_world = landmark_df['coords'][idx]
      if is_world_coordinate_valid(landmark_world):
        valid_landmark_idx.append(idx)

    image_size = landmark_mask.GetSize()
    if len(valid_landmark_idx) > 0:
      for _ in range(self.num_pos_patches_per_image):
        landmark_idx = np.random.randint(0, len(valid_landmark_idx))
        landmark_world = landmark_df['coords'][landmark_idx]
        landmark_voxel = list(landmark_mask.TransformPhysicalPointToIndex(landmark_world))
        if is_voxel_coordinate_valid(landmark_voxel, image_size):
          perturbs_idx = np.random.randint(0, len(self.positive_perturbs))
          for id in range(3):
            landmark_voxel[id] += self.positive_perturbs[perturbs_idx][id]
            landmark_voxel[id] = max(0, min(landmark_voxel[id], image_size[id] - 1))
            sample_voxels.append(landmark_voxel)

    # set negative samples, which are randomly selected from the background voxels.
    if self.num_neg_patches_per_image > 0:
      sample_neg_positions = select_random_voxels_in_multi_class_mask(
        landmark_mask, self.num_neg_patches_per_image, 0
      )
      sample_voxels.extend(sample_neg_positions)

    # assign values to the sampled area
    for voxel in sample_voxels:
      voxel_sp_x = max(0, voxel[0] - self.sampling_size[0] // 2)
      voxel_ep_x = min(image_size[0] - 1, voxel_sp_x + self.sampling_size[0])
      voxel_sp_y = max(0, voxel[1] - self.sampling_size[1] // 2)
      voxel_ep_y = min(image_size[1] - 1, voxel_sp_y + self.sampling_size[1])
      voxel_sp_z = max(0, voxel[2] - self.sampling_size[2] // 2)
      voxel_ep_z = min(image_size[2] - 1, voxel_sp_z + self.sampling_size[2])
      selected_mask_npy[voxel_sp_z:voxel_ep_z, voxel_sp_y:voxel_ep_y, voxel_sp_x:voxel_ep_x] = \
        mask_npy[voxel_sp_z:voxel_ep_z, voxel_sp_y:voxel_ep_y, voxel_sp_x:voxel_ep_x]

    # reorder the landmark label
    reordered_selected_mask_npy = np.zeros_like(selected_mask_npy) - 1.0
    reordered_selected_mask_npy[abs(selected_mask_npy) < 1e-1] = 0
    for idx in range(len(landmark_df['label'])):
      label = landmark_df['label'][idx]
      reordered_selected_mask_npy[abs(selected_mask_npy - label) < 1e-1] = idx + 1

    reordered_selected_mask = sitk.GetImageFromArray(reordered_selected_mask_npy) # This makes the mask as nifti image
    reordered_selected_mask.CopyInformation(landmark_mask) #Copies Physical world space such as origin , spacing etc.

    return reordered_selected_mask

  def build_transform(self, cfg):
      transforms = []

      if cfg.affine_turn_on:
          transforms.append(
              tio.RandomAffine(
                  scales=cfg.scales,
                  degrees=cfg.rotation,
                  translation=cfg.translation,
                  p=cfg.affine_p
              )
          )

      if cfg.flip_turn_on:
          transforms.append(
              tio.RandomFlip(
                  axes=0,
                  p=cfg.flip_p
              )
          )

      if cfg.elastic_turn_on:
          transforms.append(
              tio.RandomElasticDeformation(
                  num_control_points=cfg.elastic_num_control_points,
                  max_displacement=cfg.elastic_max_displacement,
                  locked_borders=cfg.elastic_locked_borders,
                  include=('image',),
                  p=cfg.elastic_p
              )
          )

      if cfg.motion_turn_on:
          transforms.append(
              tio.RandomMotion(
                  num_transforms=cfg.motion_num_transforms,
                  include=('image',),
                  p=cfg.motion_p
              )
          )

      if cfg.noise_turn_on:
          transforms.append(
              tio.RandomNoise(
                  mean=cfg.noise_mean,
                  std=cfg.noise_std,
                  include=('image',),
                  p=cfg.noise_p
              )
          )

      if cfg.gamma_turn_on:
          transforms.append(
              tio.RandomGamma(
                  log_gamma=cfg.log_gamma,
                  include=('image',),
                  p=cfg.gamma_p
              )
          )

      return tio.Compose(transforms)
  
  def augment(self, image, mask=None):
      """
      Apply TorchIO augmentations to a 3D CBCT image and optional mask.
      
      Args:
          image (SimpleITK.Image): Input cropped image in physical space.
          mask (SimpleITK.Image, optional): Corresponding label/landmark mask.
      
      Returns:
          (image_aug, mask_aug) if mask is provided, else image_aug only
      """

      # Convert to TorchIO tensors (C, H, W, D) format
      image_tensor = sitk.GetArrayFromImage(image).astype(np.float32)[None]
      subject_dict = {
          'image': tio.ScalarImage(tensor=image_tensor)
      }

      if mask is not None:
          mask_tensor = sitk.GetArrayFromImage(mask).astype(np.float32)[None]
          subject_dict['mask'] = tio.LabelMap(tensor=mask_tensor)

      subject = tio.Subject(subject_dict)

      # Apply the composed transform
      transformed = self.aug_transform(subject)

      # Convert back to SimpleITK image
      image_aug = sitk.GetImageFromArray(transformed['image'].data.squeeze(0).numpy())
      image_aug.CopyInformation(image)

      if mask is not None:
          mask_aug = sitk.GetImageFromArray(transformed['mask'].data.squeeze(0).numpy())
          mask_aug.CopyInformation(mask)
          return image_aug, mask_aug

      return image_aug


  def __getitem__(self, index):
    """ get a training sample - image(s) and segmentation pair
    :param index:  the sample index
    :return cropped image, cropped mask, crop frame, case name
    """
    # image IO
    image_name = self.image_name_list[index]
    image_path = self.image_path_list[index]

    images = []
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    images.append(image)

    landmark_coords = self.landmark_coords_dict[image_name]
    landmark_mask_path = self.landmark_mask_path[index]
    landmark_mask = sitk.ReadImage(landmark_mask_path, sitk.sitkFloat32)
    
    image = pad_image(image, self.pad_size, pad_value=0)
    landmark_mask = pad_image(landmark_mask, self.pad_size, pad_value=-1)

    # sampling a crop center
    if self.sampling_method == 'GLOBAL':
      crop_center_mm = self.global_sample(landmark_mask) # From landmark_mask find a random point in mm.

    else:
      raise ValueError('Only support CENTER, GLOBAL, MASK, and HYBRID sampling methods')

    # random translation
    crop_center_mm += np.random.uniform(-self.augmentation_translation_lmk, self.augmentation_translation_lmk, size=[3])

    # sample a crop from image and normalize it
    # We are working with one image and one mask at a time.
    
    landmark_mask = crop_image(landmark_mask, crop_center_mm, self.crop_size, self.crop_spacing, 'NN')
    landmark_mask = self.select_samples_in_the_landmark_mask(landmark_mask, self.landmark_coords_dict[image_name])

    for idx in range(len(images)):
      images[idx] = crop_image(images[idx], crop_center_mm, self.crop_size, self.crop_spacing, self.interpolation)
      if self.crop_normalizers[idx] is not None:
          images[idx] = self.crop_normalizers[idx](images[idx])

      # Apply augmentations
      if self.mode == 'train' and self.augmentation_turn_on:
          images[idx], landmark_mask = self.augment(images[idx], landmark_mask)

    # convert image and masks to tensors
    image_tensor = convert_image_to_tensor(images)
    landmark_mask_tensor = convert_image_to_tensor(landmark_mask)

    # convert landmark coords to tensor
    landmark_coords_list = []
    indices = np.argsort(landmark_coords['label'])
    for idx in indices:
      coords = landmark_coords['coords'][idx]
      landmark_coords_list.append([coords[0], coords[1], coords[2]])
    landmark_coords_tensor = torch.from_numpy(np.array(landmark_coords_list, dtype=np.float32))

    # image frame
    image_frame = get_image_frame(images[0])

    return image_tensor, landmark_mask_tensor, \
            landmark_coords_tensor, image_frame, image_name