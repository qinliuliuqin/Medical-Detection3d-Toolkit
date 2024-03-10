import numpy as np
import pandas as pd
import SimpleITK as sitk

from detection3d.utils.image_tools import resample_spacing
from detection3d.utils.landmark_utils import is_voxel_coordinate_valid


def gen_single_landmark_mask(
    ref_image: sitk.Image, 
    landmark_df: pd.DataFrame,
    landmark_label, 
    spacing, 
    pos_upper_bound, 
    neg_lower_bound
):
    """
    Generate landmark mask for a single image.
    """
    ref_image = resample_spacing(ref_image, spacing, 1, 'LINEAR')
    ref_image_npy = sitk.GetArrayFromImage(ref_image)
    ref_image_size = ref_image.GetSize()
    landmark_mask_npy = np.zeros_like(ref_image_npy).astype(np.int16)

    for _, row in landmark_df.iterrows():
        world = [row['x'], row['y'], row['z']]
        voxel = ref_image.TransformPhysicalPointToIndex(world)
        for x in range(voxel[0] - neg_lower_bound, voxel[0] + neg_lower_bound):
            for y in range(voxel[1] - neg_lower_bound, voxel[1] + neg_lower_bound):
                for z in range(voxel[2] - neg_lower_bound, voxel[2] + neg_lower_bound):
                    if is_voxel_coordinate_valid([x, y, z], ref_image_size):
                        distance = np.linalg.norm(np.array([x, y, z], dtype=np.float32) - voxel)
                        if distance < pos_upper_bound:
                            landmark_mask_npy[z, y, x] = int(landmark_label[row['name']])
                        elif distance < neg_lower_bound and abs(landmark_mask_npy[z, y, x]) <= 0:
                            landmark_mask_npy[z, y, x] = -1

    landmark_mask = sitk.GetImageFromArray(landmark_mask_npy)
    landmark_mask.CopyInformation(ref_image)

    return landmark_mask