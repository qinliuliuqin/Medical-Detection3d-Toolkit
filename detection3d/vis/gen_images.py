from collections import namedtuple, OrderedDict
import numpy as np
import os
import pandas as pd
from scipy import interpolate as itp
import SimpleITK as sitk

from detection3d.vis.save_images import PlotCrossingAndSavePlanes, save_black_planes
from detection3d.utils.landmark_utils import is_world_coordinate_valid, is_voxel_coordinate_valid


"""
The struct containing the 2D plane generation options.
"""
GenImagesOptions = namedtuple('GenImagesOption',
                              'resolution contrast_min contrast_max')


def get_landmarks_stat(landmarks):
  """
  Get the statistics of landmarks.
  """
  landmarks_stat = OrderedDict()
  for landmark_name in landmarks[list(landmarks.keys())[0]].keys():
    landmarks_stat.update({landmark_name: {'pos': [], 'neg': []}})

  for image_name in landmarks.keys():
    landmark_names = landmarks[image_name].keys()
    image_baseame = image_name.split('/')[0]
    for landmark_name in landmark_names:
      landmark = landmarks[image_name][landmark_name]
      if is_world_coordinate_valid(landmark):
        landmarks_stat[landmark_name]['pos'].append(image_baseame)
      else:
        landmarks_stat[landmark_name]['neg'].append(image_baseame)

  return landmarks_stat


def load_coordinates_from_csv(csv_file):
  """
  Load coordinates (x,y,z) from CSV file and pack them into an OrderedDict.
  Input arguments:
    csv_file: CSV file path containing [x,y,z] coordinates in each row.
  Return:
    Two dimensional array, {idx_0:[x_0, y_0, z_0], ... ,idx_n:[x_n, y_n, z_n]},
    where idx_n is the landmark name.
  """
  assert csv_file.endswith('.csv')
  
  landmarks = OrderedDict()
  df = pd.read_csv(csv_file)
  for idx in range(len(df)):
    landmark_name = df['name'][idx]
    x, y, z = df['x'][idx], df['y'][idx], df['z'][idx]
    landmarks.update({landmark_name: [x, y, z]})
  
  return landmarks


def extract_planes_from_volume(src_data, src_dimension, src_resolution,
                               dst_dimension, dst_resolution, dst_point):
  """
  Extract axial, coronal, and sagittal planes crossing the input point from a
  3D volume data. The three planes are resampled to the specified resolution.

  Input arguments:
    src_data:       The source 3D data volume.
    src_dimension:  Three dimensional vector of the data size.
    src_resolution: Three or one (isotropic) dimensional vector of the data resolution.
    dst_dimension:  Three dimensional vector of the resampled data size.
    dst_resolution: Three or one (isotropic) dimensional vector of the resampled
                    data resolution.
    dst_point:      The point coordinate in the resampled data space.
  Return:
    dst_axial:      Extracted (resampled) 2-D axial plane crossing the dst_point.
    dst_coronal:    Extracted (resampled) 2-D coronal plane crossing the dst_point.
    dst_sagittal:   Extracted (resampled) 2-D sagittal plane crossing the dst_point.
  """
  src_x = np.arange(src_dimension[0])
  src_y = np.arange(src_dimension[1])
  src_z = np.arange(src_dimension[2])

  dst_x = np.linspace(0, src_dimension[0] - 1, dst_dimension[0])
  dst_y = np.linspace(0, src_dimension[1] - 1, dst_dimension[1])
  dst_z = np.linspace(0, src_dimension[2] - 1, dst_dimension[2])

  # Extract the axial plane from the source data passing through the dst point.
  # meshgrid() function scans the volume along Y-axis first, then X-axis,
  # then Z-axis. Hence, we switch the order of y and x to make sure the
  # grid point coordinates are in correct order of (x,y,z).
  dst_axial_y, dst_axial_x, dst_axial_z = np.meshgrid(
    dst_y, dst_x, [min(dst_point[2] * dst_resolution[2] / src_resolution[2], src_dimension[2] - 1)])

  dst_axial_interp = itp.interpn(
    (src_x, src_y, src_z), src_data, (dst_axial_x, dst_axial_y, dst_axial_z))

  dst_axial = dst_axial_interp[:,:,0]

  # Extract the coronal plane from the source data passing through the dst point.
  dst_coronal_y, dst_coronal_x, dst_coronal_z = np.meshgrid(
    [min(dst_point[1] * dst_resolution[1] / src_resolution[1], src_dimension[1] - 1)], dst_x, dst_z)

  dst_coronal_interp = itp.interpn(
    (src_x, src_y, src_z), src_data, (dst_coronal_x, dst_coronal_y, dst_coronal_z))

  dst_coronal = dst_coronal_interp[:,0,:]

  # Extract the sagittal plane from the source data passing through the dst point.
  dst_sagittal_y, dst_sagittal_x, dst_sagittal_z = np.meshgrid(
    dst_y, [min(dst_point[0] * dst_resolution[0] / src_resolution[0], src_dimension[0] - 1)], dst_z)

  dst_sagittal_interp = itp.interpn(
    (src_x, src_y, src_z), src_data, (dst_sagittal_x, dst_sagittal_y, dst_sagittal_z))

  dst_sagittal = dst_sagittal_interp[0,:,:]

  return dst_axial, dst_coronal, dst_sagittal


def gen_plane_images(image_folder, landmarks, image_type, output_contrast_range,
                     output_image_spacing, output_folder):
  image_name_list = list(landmarks.keys())
  image_name_list.sort()
  for idx, image_name in enumerate(image_name_list):
    print("Generate plane images for {}.".format(image_name))
    assert len(landmarks[image_name].keys()) > 0
    
    src_image = sitk.ReadImage(os.path.join(image_folder, image_name, 'org.mha'))
    src_size, src_spacing = src_image.GetSize(), src_image.GetSpacing()
    src_image_npy = sitk.GetArrayFromImage(src_image)
    src_image_npy = np.asarray(src_image_npy.astype(np.int16), dtype=np.int16).transpose([2, 1, 0])

    # resample to the desired spacing
    dst_spacing = np.array(output_image_spacing, dtype=np.float32)
    dst_size = np.zeros(3, dtype=int)
    for i in range(3):
      dst_size[i] = int(np.ceil(src_size[i] * src_spacing[i] / dst_spacing[i]))

    for landmark_idx, landmark_name in enumerate(landmarks[image_name].keys()):
      landmark_output_folder = os.path.join(output_folder, 'lm{}'.format(landmark_idx), 'pictures')
      if not os.path.isdir(landmark_output_folder):
        os.makedirs(landmark_output_folder)
        
      image_basename = image_name.split('/')[0]
      axial_image_filename = "{0}_{1}_lm{2}_axial.png".format(image_basename, image_type, landmark_name)
      coronal_image_filename = "{0}_{1}_lm{2}_coronal.png".format(image_basename, image_type, landmark_name)
      sagittal_image_filename = "{0}_{1}_lm{2}_sagittal.png".format(image_basename, image_type, landmark_name)

      src_landmark_coord_world = landmarks[image_name][landmark_name]
      src_landmark_coord_world_double = [float(src_landmark_coord_world[idx]) for idx in range(3)]
      src_landmark_coord_voxel = src_image.TransformPhysicalPointToContinuousIndex(src_landmark_coord_world_double)
      if not is_world_coordinate_valid(src_landmark_coord_world) or \
          not is_voxel_coordinate_valid(src_landmark_coord_voxel, src_size):
        save_black_planes('axial', os.path.join(landmark_output_folder, axial_image_filename), 'g')
        save_black_planes('coronal', os.path.join(landmark_output_folder, coronal_image_filename), 'r')
        save_black_planes('sagittal', os.path.join(landmark_output_folder, sagittal_image_filename), 'y')
        continue
        
      dst_landmark_coord_voxel = np.zeros(3, dtype=int)
      for i in range(3):
        dst_landmark_coord_voxel[i] = int(np.floor(src_landmark_coord_voxel[i] * src_spacing[i] / dst_spacing[i]))

      axial_plane, coronal_plane, sagittal_plane = extract_planes_from_volume(
        src_image_npy, src_size, src_spacing, dst_size, dst_spacing, dst_landmark_coord_voxel)

      if output_contrast_range is not None:
        contrast_min, contrast_max = output_contrast_range[0], output_contrast_range[1]
      else:
        axial_plane_mean, axial_plane_std = np.mean(axial_plane), np.std(axial_plane)
        coronal_plane_mean, coronal_plane_std = np.mean(coronal_plane), np.std(coronal_plane)
        sagittal_plane_mean, sagittal_plane_std = np.mean(sagittal_plane), np.std(sagittal_plane)
        axial_plane = (axial_plane - axial_plane_mean) / axial_plane_std
        coronal_plane = (coronal_plane - coronal_plane_mean) / coronal_plane_std
        sagittal_plane = (sagittal_plane - sagittal_plane_mean) / sagittal_plane_std
        contrast_min, contrast_max = 0, 2
        
      axial_plane = np.clip(axial_plane, contrast_min, contrast_max)
      coronal_plane = np.clip(coronal_plane, contrast_min, contrast_max)
      sagittal_plane = np.clip(sagittal_plane, contrast_min, contrast_max)

      PlotCrossingAndSavePlanes(
        axial_plane, [dst_landmark_coord_voxel[0], dst_landmark_coord_voxel[1]],
        'axial', os.path.join(landmark_output_folder, axial_image_filename), 'g', 'g-.')

      PlotCrossingAndSavePlanes(
        coronal_plane, [dst_landmark_coord_voxel[0], dst_landmark_coord_voxel[2]],
        'coronal', os.path.join(landmark_output_folder, coronal_image_filename), 'r', 'r-.')

      PlotCrossingAndSavePlanes(
        sagittal_plane, [dst_landmark_coord_voxel[1], dst_landmark_coord_voxel[2]],
        'sagittal', os.path.join(landmark_output_folder, sagittal_image_filename), 'y', 'y-.')
