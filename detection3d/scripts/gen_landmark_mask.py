from __future__ import print_function
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd

from detection3d.utils.image_tools import resample_spacing
from detection3d.utils.landmark_utils import is_world_coordinate_valid, is_voxel_coordinate_valid


def gen_single_landmark_mask(ref_image, landmark_df, spacing, pos_upper_bound, neg_lower_bound):
  assert isinstance(ref_image, sitk.Image)

  ref_image = resample_spacing(ref_image, spacing, 1, 'NN')
  ref_image_npy = sitk.GetArrayFromImage(ref_image)
  ref_image_size = ref_image.GetSize()
  landmark_mask_npy = np.zeros_like(ref_image_npy)
  for landmark_name in landmark_df.keys():
    landmark_label = landmark_df[landmark_name]['label']
    landmark_world = [landmark_df[landmark_name]['x'],
                      landmark_df[landmark_name]['y'],
                      landmark_df[landmark_name]['z']]
    landmark_voxel = ref_image.TransformPhysicalPointToIndex(landmark_world)
    for x in range(landmark_voxel[0] - neg_lower_bound,
                   landmark_voxel[0] + neg_lower_bound):
      for y in range(landmark_voxel[1] - neg_lower_bound,
                     landmark_voxel[1] + neg_lower_bound):
        for z in range(landmark_voxel[2] - neg_lower_bound,
                       landmark_voxel[2] + neg_lower_bound):
          if is_voxel_coordinate_valid([x, y, z], ref_image_size):
            distance = np.linalg.norm(np.array([x, y, z], dtype=np.float32) - landmark_voxel)
            if distance < pos_upper_bound:
              landmark_mask_npy[z, y, x] = float(landmark_label)
            elif distance < neg_lower_bound and abs(landmark_mask_npy[z, y, x]) < 1e-6:
              landmark_mask_npy[z, y, x] = -1.0

  landmark_mask = sitk.GetImageFromArray(landmark_mask_npy)
  landmark_mask.CopyInformation(ref_image)

  return landmark_mask


def gen_landmark_mask_batch(image_folder, landmark_folder, target_landmark_label,
                            spacing, pos_upper_bound, neg_lower_bound, landmark_mask_save_folder):

  # get image name list
  landmark_files = os.listdir(landmark_folder)
  image_names = []
  for landmark_file in landmark_files:
    if landmark_file.startswith('case'):
      image_names.append(landmark_file.split('.')[0])
  image_names.sort()

  if not os.path.isdir(landmark_mask_save_folder):
    os.makedirs(landmark_mask_save_folder)

  for image_name in image_names:
    print(image_name)
    landmark_df = pd.read_csv(os.path.join(landmark_folder, '{}.csv'.format(image_name)))
    target_landmark_df = {}
    for landmark_name in target_landmark_label.keys():
      landmark_label = target_landmark_label[landmark_name]
      x = landmark_df[landmark_df['name'] == landmark_name]['x'].values[0]
      y = landmark_df[landmark_df['name'] == landmark_name]['y'].values[0]
      z = landmark_df[landmark_df['name'] == landmark_name]['z'].values[0]
      if is_world_coordinate_valid([x, y, z]):
        target_landmark_df[landmark_name] = {}
        target_landmark_df[landmark_name]['label'] = landmark_label
        target_landmark_df[landmark_name]['x'] = float(x)
        target_landmark_df[landmark_name]['y'] = float(y)
        target_landmark_df[landmark_name]['z'] = float(z)

    image = sitk.ReadImage(os.path.join(image_folder, image_name, 'org.mha'))
    landmark_mask = gen_single_landmark_mask(
      image, target_landmark_df, spacing, pos_upper_bound, neg_lower_bound
    )

    sitk.WriteImage(landmark_mask, os.path.join(landmark_mask_save_folder, '{}.mha'.format(image_name)), True)


def gen_landmark_2mm():
  batch_idx = 3
  image_folder = '/mnt/projects/CT_Dental/data'
  landmark_folder = '/mnt/projects/CT_Dental/landmark'
  landmark_mask_save_folder = '/mnt/projects/CT_Dental/landmark_mask/batch_{}_2.0mm'.format(batch_idx)
  landmark_label_file = '/home/ql/projects/dental_image_analysis/detection/scripts/batch_{}_2mm.csv'.format(batch_idx)
  spacing = [2.0, 2.0, 2.0]  # mm
  pos_upper_bound = 3  # voxel
  neg_lower_bound = 6  # voxel

  landmark_label_df = pd.read_csv(landmark_label_file)
  target_landmark_label = {}
  for row in landmark_label_df.iterrows():
    target_landmark_label.update({row[1]['landmark_name']: row[1]['landmark_label']})

  gen_landmark_mask_batch(image_folder, landmark_folder, target_landmark_label, spacing,
                          pos_upper_bound, neg_lower_bound, landmark_mask_save_folder)



if __name__ == '__main__':

  steps = [1]

  if 1 in steps:
    gen_landmark_2mm()
