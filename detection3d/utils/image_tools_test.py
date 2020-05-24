import SimpleITK as sitk

from detection.utils.image_tools import mask_to_mesh
from segmentation3d.utils.image_tools import resample_spacing


def test_mask_to_mesh():
    mask_path_mha = '/mnt/projects/CT_Dental/test_seg.mha'
    mask_path_nii = '/mnt/projects/CT_Dental/test_seg.nii.gz'

    stl_path = '/mnt/projects/CT_Dental/test_seg.stl'
    label = 1
    mask_to_mesh(mask_path_mha, stl_path, label)


def resample_image():
    image_path = '/mnt/projects/CT_Dental/data/case_174_ct_normal/org.mha'
    image = sitk.ReadImage(image_path)
    resampled_image = resample_spacing(image, [0.4, 0.4, 0.4], 1, 'LINEAR')
    resampled_image_path = '/mnt/projects/CT_Dental/landmark_mask/batch_1_0.4mm/case_174_ct_normal_org.mha'
    sitk.WriteImage(resampled_image, resampled_image_path)

    mask_path = '/mnt/projects/CT_Dental/data/case_174_ct_normal/seg.mha'
    mask = sitk.ReadImage(mask_path)
    resampled_mask = resample_spacing(mask, [0.4, 0.4, 0.4], 1, 'NN')
    resampled_mask_path = '/mnt/projects/CT_Dental/landmark_mask/batch_1_0.4mm/case_174_ct_normal_seg.mha'
    sitk.WriteImage(resampled_mask, resampled_mask_path)


if __name__ == '__main__':

    # test_mask_to_mesh()

    resample_image()