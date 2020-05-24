import numpy as np
import pandas as pd


def is_voxel_coordinate_valid(coord_voxel, image_size):
    """
    Check whether the voxel coordinate is out of bound.
    """
    for idx in range(3):
        if coord_voxel[idx] < 0 or coord_voxel[idx] >= image_size[idx]:
            return False
    return True


def is_world_coordinate_valid(coord_world):
    """
    Check whether the world coordinate is valid.
    The world coordinate is invalid if it is (0, 0, 0), (1, 1, 1), or (-1, -1, -1).
    """
    coord_world_npy = np.array(coord_world)

    if np.linalg.norm(coord_world_npy, ord=1) < 1e-6 or \
            np.linalg.norm(coord_world_npy - np.ones(3), ord=1) < 1e-6 or \
            np.linalg.norm(coord_world_npy - -1 * np.ones(3), ord=1) < 1e-6:
        return False

    return True


def merge_landmark_files(landmark_files, merged_landmark_file):
    """
    Merge multiple landmark files into a single landmark file
    :param landmark_files:
    :param merged_landmark_file:
    :return: None
    """
    assert isinstance(landmark_files, list)
    assert merged_landmark_file.endswith('.csv')

    if len(landmark_files) == 0:
        return

    landmark_df = pd.read_csv(landmark_files[0])
    for idx in range(1, len(landmark_files)):
        landmark_df_temp = pd.read_csv(landmark_files[idx])
        landmark_df = pd.concat([landmark_df, landmark_df_temp], axis=0)

    landmark_df.sort_values(by=['name'], inplace=True)
    landmark_df.to_csv(merged_landmark_file, index=False)


def merge_landmark_dataframes(landmark_dataframes):
    """
    Merge multiple landmark dataframes into a single landmark dataframe
    :param landmark_dataframes: the list containing multiple dataframes
    :return: the merged dataframe
    """

    merged_landmark_df = landmark_dataframes[0]
    for idx in range(1, len(landmark_dataframes)):
        merged_landmark_df = pd.concat([merged_landmark_df, landmark_dataframes[idx]], axis=0)

    return merged_landmark_df
