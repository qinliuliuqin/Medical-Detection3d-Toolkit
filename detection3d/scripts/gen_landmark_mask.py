import argparse
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd

from detection3d.utils.image_tools import resample_spacing
from detection3d.utils.landmark_utils import is_world_coordinate_valid, is_voxel_coordinate_valid


def gen_single_landmark_mask(ref_image, landmark_df, spacing, pos_upper_bound, neg_lower_bound):
  """
  Generate landmark mask for a single image.
  """
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
  """
  Generate landmark mask for a batch of images
  """
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

    image = sitk.ReadImage(os.path.join(image_folder, image_name, 'org.nii.gz'))
    landmark_mask = gen_single_landmark_mask(
      image, target_landmark_df, spacing, pos_upper_bound, neg_lower_bound
    )

    sitk.WriteImage(landmark_mask, os.path.join(landmark_mask_save_folder, '{}.nii.gz'.format(image_name)), True)


def generate_landmark_mask(image_folder, landmark_folder, landmark_label_file, spacing, bound, save_folder):
  """
  Generate landmark mask.
  """
  landmark_label_df = pd.read_csv(landmark_label_file)
  target_landmark_label = {}
  for row in landmark_label_df.iterrows():
    target_landmark_label.update({row[1]['landmark_name']: row[1]['landmark_label']})

  pos_upper_bound, neg_lower_bound = bound[0], bound[1]
  gen_landmark_mask_batch(image_folder, landmark_folder, target_landmark_label, spacing,
                          pos_upper_bound, neg_lower_bound, save_folder)


def main():
  long_description = 'Generate landmark mask for landmark detection.'

  default_batch_idx = 5
  default_input = '/mnt/projects/CT_Dental/data_v2'
  default_landmark = '/mnt/projects/CT_Dental/landmark_v2'
  default_output = '/mnt/projects/CT_Dental/landmark_mask_v2/batch_{}_1.5mm'.format(default_batch_idx)
  default_label = '/home/ql/projects/Medical-Detection3d-Toolkit/detection3d/scripts/landmark_label_file_batch_{}.csv'.format(default_batch_idx)
  default_spacing = [1.5, 1.5, 1.5]
  default_pos_upper_bound = 3
  default_neg_lower_bound = 6

  parser = argparse.ArgumentParser(description=long_description)
  parser.add_argument('-i', '--input', default=default_input,
                      help='input folder for intensity images.')
  parser.add_argument('-l', '--landmark', default=default_landmark,
                      help='landmark folder.')
  parser.add_argument('-o', '--output', default=default_output,
                      help='output folder for the landmark mask')
  parser.add_argument('-n', '--label', default=default_label,
                      help='the label file containing the selected landmark names.')
  parser.add_argument('-s', '--spacing', default=default_spacing,
                      help='the spacing of the landmark mask.')
  parser.add_argument('-b', '--bound', default=[default_pos_upper_bound, default_neg_lower_bound],
                      help='the pos. upper bound and the neg. lower bound of the landmark mask.')

  args = parser.parse_args()
  generate_landmark_mask(args.input, args.landmark, args.label, args.spacing, args.bound, args.output)


if __name__ == '__main__':

  main()