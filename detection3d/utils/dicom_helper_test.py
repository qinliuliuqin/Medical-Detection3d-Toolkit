import numpy as np
import SimpleITK as sitk

from segmentation3d.utils.dicom_helper import read_dicom_series, write_dicom_series, write_binary_dicom_series, \
  dicom_tags_dict


def test_save_dicom_series():
  # read mha image
  seg_path = '/mnt/projects/CT_Dental/Pre_Post_Facial_Data-Ma/original_images/n03_orginImg_post.nii.gz'
  seg = sitk.ReadImage(seg_path, sitk.sitkInt16)

  # save mha to dicom series
  tags = dicom_tags_dict()
  dicom_save_folder = '/mnt/projects/CT_Dental/Pre_Post_Facial_Data-Ma/original_images_dicom_test/n03_orginImg_post'
  write_dicom_series(seg, dicom_save_folder, tags=tags)

  # load the saved dicom series
  seg_reload = read_dicom_series(dicom_save_folder)
  seg_reload_path = '/mnt/projects/CT_Dental/Pre_Post_Facial_Data-Ma/original_images_dicom_test/n03_orginImg_post.nii.gz'
  sitk.WriteImage(seg_reload, seg_reload_path)

  # compare the original image and the reloaded image
  image_npy = sitk.GetArrayFromImage(seg)
  image_reloaded_npy = sitk.GetArrayFromImage(seg_reload)
  assert np.sum(np.abs(image_npy - image_reloaded_npy)) < 1e-6


def test_save_binary_dicom_series():
  # read mha image
  seg_path = '/home/qinliu/debug/seg.mha'
  seg = sitk.ReadImage(seg_path, sitk.sitkInt16)

  # save mha to binary dicom series
  tags = dicom_tags_dict()
  dicom_save_folder = '/home/qinliu/debug/seg_dicom_maxilla'
  write_binary_dicom_series(seg, dicom_save_folder, in_label=1, out_label=100, tags=tags)

  dicom_save_folder = '/home/qinliu/debug/seg_dicom_mandible'
  write_binary_dicom_series(seg, dicom_save_folder, in_label=2, out_label=100, tags=tags)


def test_merge_mask():
  """
  Merge two masks.
  mask1: a mask which includes midface, mandible and soft tissue
  mask2: a mask which includes the upper teeth and the lower teeth. The upper teeth is a part of midface, and the lower teeth
         is a part of mandible.
  merged_mask: the mask merged by mask1 and mask2.
  """
  # read the first mask
  mask_path_1 = '/home/qinliu/debug/seg1_dicom'
  mask1 = read_dicom_series(mask_path_1)

  # read the second mask
  mask_path_2 = '/home/qinliu/debug/seg2_dicom'
  mask2 = read_dicom_series(mask_path_2)

  # the two masks should have the same size
  size_mask_1, size_mask_2 = mask1.GetSize(), mask2.GetSize()
  assert size_mask_1[0] == size_mask_2[0]
  assert size_mask_1[1] == size_mask_2[1]
  assert size_mask_1[2] == size_mask_2[2]

  # merge two masks
  mask1_npy = sitk.GetArrayFromImage(mask1)
  mask2_npy = sitk.GetArrayFromImage(mask2)

  upper_teeth_label, lower_teeth_label = 1, 2
  mask1_npy[mask2_npy == upper_teeth_label] = upper_teeth_label
  mask1_npy[mask2_npy == lower_teeth_label] = lower_teeth_label

  merged_mask = sitk.GetImageFromArray(mask1_npy)
  merged_mask.CopyInformation(mask1)
  merged_mask_path = '/home/qinliu/debug/merged_seg.mha'
  sitk.WriteImage(merged_mask, merged_mask_path, True)


if __name__ == '__main__':

  test_save_dicom_series()

  # test_save_binary_dicom_series()
  #
  # test_merge_mask()