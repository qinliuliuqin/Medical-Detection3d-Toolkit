import argparse
import os
import SimpleITK as sitk
import time

from detection3d.utils.image_tools import crop_image
from detection3d.core.lmk_det_infer import read_test_folder, detection_single_image, load_det_model
from detection3d.utils.landmark_utils import merge_landmark_dataframes
from detection3d.dataset.landmark_dataset import read_image_list
from detection3d.utils.landmark_utils import is_voxel_coordinate_valid, is_world_coordinate_valid

# Note: landmarks L6CF-R, L6CF-L, L6DC-R, L6DC-L are not detected.
def dental_detection_batch(input_path, model_folder, gpu_id, output_folder):
    """

    :param input_path:
    :return:
    """
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

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # test each case
    for i, file_path in enumerate(file_path_list):
        print('{}: {}'.format(i, file_path))

        if not os.path.isfile(file_path):
            print('File {} does not exist!'.format(file_path))
            continue

        output_csv_file = os.path.join(output_folder, '{}.csv'.format(file_name_list[i]))
        begin = time.time()
        dental_detection(file_path, model_folder, gpu_id, output_csv_file)
        prediction_time = time.time() - begin

        print('Prediction: {:.2f} s'.format(prediction_time))



def dental_detection(input, model_folder, gpu_id, output_csv_file):
    """

    :param input:
    :param model_folder:
    :param gpu_id:
    :param output_csv_file:
    :return:
    """
    assert output_csv_file.endswith('.csv')

    image = sitk.ReadImage(input, sitk.sitkFloat32)
    image_name = os.path.basename(input)

    landmark_dataframes = []

    # detect batch 1
    print('Start detecting the non-teeth landmarks ...')
    batch_1_model = load_det_model(os.path.join(model_folder, 'batch_1'))
    landmark_batch_1 = detection_single_image(image, image_name, batch_1_model, gpu_id, None, None)
    del batch_1_model
    landmark_dataframes.append(landmark_batch_1)

    # detect batch 2
    batch_2_model = load_det_model(os.path.join(model_folder, 'batch_2'))
    landmark_batch_2 = detection_single_image(image, image_name, batch_2_model, gpu_id, None, None)
    del batch_2_model
    landmark_dataframes.append(landmark_batch_2)

    # detect batch 3
    batch_3_model = load_det_model(os.path.join(model_folder, 'batch_3'))
    landmark_batch_3 = detection_single_image(image, image_name, batch_3_model, gpu_id, None, None)
    del batch_3_model
    landmark_dataframes.append(landmark_batch_3)

    # crop the teeth region according to landmark 'L0'
    print('Start detecting the teeth landmarks')
    l0 = landmark_batch_3[landmark_batch_3['name'] == 'L0']
    world_coord_l0 = [l0['x'].values[0], l0['y'].values[0], l0['z'].values[0]]
    if is_world_coordinate_valid(world_coord_l0):
        voxel_coord_l0 = image.TransformPhysicalPointToIndex(world_coord_l0)
        if is_voxel_coordinate_valid(voxel_coord_l0, image.GetSize()):
            image_spacing, image_size = image.GetSpacing(), image.GetSize()
            crop_spacing, crop_size = [0.8, 0.8, 0.8], [128, 96, 96]
            resample_ratio = [crop_spacing[idx] / image_spacing[idx] for idx in range(3)]

            offset = [-int(64 * resample_ratio[0]), -int(16 * resample_ratio[1]), -int(48 * resample_ratio[2])]
            left_bottom_voxel = [voxel_coord_l0[idx] + offset[idx] for idx in range(3)]
            right_top_voxel = [left_bottom_voxel[idx] + int(crop_size[idx] * resample_ratio[idx]) - 1 for idx in range(3)]
            for idx in range(3):
                left_bottom_voxel[idx] = max(0, left_bottom_voxel[idx])
                right_top_voxel[idx] = min(image_size[idx] - 1, right_top_voxel[idx])

            crop_voxel_center = [(left_bottom_voxel[idx] + right_top_voxel[idx]) // 2 for idx in range(3)]
            crop_world_center = image.TransformContinuousIndexToPhysicalPoint(crop_voxel_center)
            cropped_image = crop_image(image, crop_world_center, crop_size, crop_spacing, 'LINEAR')

            # detect batch 4-lower teeth batch 1
            batch_4_lower_1_model = load_det_model(os.path.join(model_folder, 'batch_4_lower_1'))
            landmark_batch_4_lower_1 = detection_single_image(cropped_image, image_name, batch_4_lower_1_model, gpu_id, None, None)
            del batch_4_lower_1_model
            landmark_dataframes.append(landmark_batch_4_lower_1)

            # detect batch 4-lower teeth batch 2
            batch_4_lower_2_model = load_det_model(os.path.join(model_folder, 'batch_4_lower_2'))
            landmark_batch_4_lower_2 = detection_single_image(cropped_image, image_name, batch_4_lower_2_model, gpu_id, None, None)
            del batch_4_lower_2_model
            landmark_dataframes.append(landmark_batch_4_lower_2)

            # detect batch 4-upper teeth batch 1
            batch_4_upper_1_model = load_det_model(os.path.join(model_folder, 'batch_4_upper_1'))
            landmark_batch_4_upper_1 = detection_single_image(cropped_image, image_name, batch_4_upper_1_model, gpu_id, None, None)
            del batch_4_upper_1_model
            landmark_dataframes.append(landmark_batch_4_upper_1)

            # detect batch 4-upper teeth batch 2
            batch_4_upper_2_model = load_det_model(os.path.join(model_folder, 'batch_4_upper_2'))
            landmark_batch_4_upper_2 = detection_single_image(cropped_image, image_name, batch_4_upper_2_model, gpu_id, None, None)
            del batch_4_upper_2_model
            landmark_dataframes.append(landmark_batch_4_upper_2)

    merged_landmark_dataframes = merge_landmark_dataframes(landmark_dataframes)
    merged_landmark_dataframes.sort_values(by=['name'], inplace=True)
    merged_landmark_dataframes.to_csv(output_csv_file, index=False)


def main():
    long_description = 'Inference engine for 3d medical image landmark detection' \

    default_input = '/shenlab/lab_stor6/projects/CT_Dental/data/case_176_ct_normal/org.mha'
    default_model = '/shenlab/lab_stor6/qinliu/projects/CT_Dental/models/model_0514_2020'
    default_output = '/shenlab/lab_stor6/qinliu/projects/CT_Dental/results/model_0514_2020/case_176_ct_normal.csv'
    default_gpu_id = 1

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', default=default_input,
                        help='input intensity images')
    parser.add_argument('-m', '--model', default=default_model,
                        help='model root folder')
    parser.add_argument('-g', '--gpu_id', type=int, default=default_gpu_id,
                        help='the gpu id to run model, set to -1 if using cpu only.')
    parser.add_argument('-o', '--output', default=default_output,
                        help='output folder for segmentation')

    args = parser.parse_args()
    dental_detection_batch(args.input, args.model, args.gpu_id, args.output)


if __name__ == '__main__':
    main()
