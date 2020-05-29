from collections import namedtuple
import numpy as np

from detection3d.vis.gen_images import get_landmarks_stat


"""
The struct containing the error summary.
"""
ErrorSummary = namedtuple('ErrorSummary',
                          'all_cases, tp_cases tn_cases fp_cases fn_cases error_dx '
                          'error_dy error_dz error_l2 error_type error_sorted_index '
                          'mean_error_tp std_error_tp median_error_tp max_error_tp')


def error_analysis(label_landmark, detection_landmark, decending=True):
  """
  Analyze landmark detection error and return the error statistics summary.
  Input arguments:
  label_landmark: A dict whose keys and values are filenames and coordinates of labelled points respectively.
  detection_landmark: A dict whose keys and values are filenames and coordinates of detected points respectively.
  descending:          Flag indicating whether errors sorted in ascending or descending order.
  Return:
  error_summary:       Summary of error statistics.
  """
  # get the true positive files
  detected_landmarks_stat = get_landmarks_stat(detection_landmark)
  labelled_landmarks_stat = get_landmarks_stat(label_landmark)
  
  tp_cases, tn_cases, fp_cases, fn_cases = {}, {}, {}, {}
  error_dx, error_dy, error_dz, error_l2 = {}, {}, {}, {}
  mean_error_tp, std_error_tp, median_error_tp, max_error_tp = {}, {}, {}, {}
  error_sorted_index, error_type, all_cases = {}, {}, {}
  for landmark_name in detected_landmarks_stat.keys():
    tp_cases_list = list(set(detected_landmarks_stat[landmark_name]['pos']) &
                    set(labelled_landmarks_stat[landmark_name]['pos']))
    tn_cases_list = list(set(detected_landmarks_stat[landmark_name]['neg']) &
                    set(labelled_landmarks_stat[landmark_name]['neg']))
    fp_cases_list = list(set(detected_landmarks_stat[landmark_name]['pos']) &
                    set(labelled_landmarks_stat[landmark_name]['neg']))
    fn_cases_list = list(set(detected_landmarks_stat[landmark_name]['neg']) &
                    set(labelled_landmarks_stat[landmark_name]['pos']))
    
    error_dx_list, error_dy_list, error_dz_list, error_l2_list = \
      [], [], [], []
    all_file_list = []
    error_type_list = []
    for file_name in tp_cases_list:
      dx = detection_landmark[file_name][landmark_name][0] - \
           label_landmark[file_name][landmark_name][0]
      error_dx_list.append(dx)
      dy = detection_landmark[file_name][landmark_name][1] - \
           label_landmark[file_name][landmark_name][1]
      error_dy_list.append(dy)
      dz = detection_landmark[file_name][landmark_name][2] - \
           label_landmark[file_name][landmark_name][2]
      error_dz_list.append(dz)
      l2 = np.linalg.norm([dx, dy, dz])
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('TP')
      
    mean_error_tp.update({landmark_name: np.mean(error_l2_list)})
    std_error_tp.update({landmark_name: np.std(error_l2_list)})
    median_error_tp.update({landmark_name: np.median(error_l2_list)})
    max_error_tp.update({landmark_name: np.max(error_l2_list)})

    for file_name in tn_cases_list:
      dx = 0
      error_dx_list.append(dx)
      dy = 0
      error_dy_list.append(dy)
      dz = 0
      error_dz_list.append(dz)
      l2 = 0
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('TN')

    for file_name in fp_cases_list:
      dx = -1
      error_dx_list.append(dx)
      dy = -1
      error_dy_list.append(dy)
      dz = -1
      error_dz_list.append(dz)
      l2 = -1
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('FP')

    for file_name in fn_cases_list:
      dx = -1
      error_dx_list.append(dx)
      dy = -1
      error_dy_list.append(dy)
      dz = -1
      error_dz_list.append(dz)
      l2 = -1
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('FN')

    all_cases.update({landmark_name: all_file_list})
    tp_cases.update({landmark_name: tp_cases_list})
    tn_cases.update({landmark_name: tn_cases_list})
    fp_cases.update({landmark_name: fp_cases_list})
    fn_cases.update({landmark_name: fn_cases_list})
    error_dx.update({landmark_name: error_dx_list})
    error_dy.update({landmark_name: error_dy_list})
    error_dz.update({landmark_name: error_dz_list})
    error_l2.update({landmark_name: error_l2_list})
    sorted_index_list = np.argsort(error_l2[landmark_name])
    if decending:
      sorted_index_list = sorted_index_list[::-1]
    error_sorted_index.update({landmark_name: sorted_index_list})
    error_type.update({landmark_name: error_type_list})

  error_summary = ErrorSummary(
    all_cases=all_cases,
    tp_cases=tp_cases,
    tn_cases=tn_cases,
    fp_cases=fp_cases,
    fn_cases=fn_cases,
    error_dx=error_dx,
    error_dy=error_dy,
    error_dz=error_dz,
    error_l2=error_l2,
    error_type=error_type,
    error_sorted_index=error_sorted_index,
    mean_error_tp=mean_error_tp,
    std_error_tp=std_error_tp,
    median_error_tp=median_error_tp,
    max_error_tp=max_error_tp
  )

  return error_summary
