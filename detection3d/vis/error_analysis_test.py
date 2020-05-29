import numpy as np
from detection3d.vis.error_analysis import error_analysis


def CheckEqual(value, expected_value, threshold = 1e-5):
  diff = np.linalg.norm(value - expected_value)
  if diff >= threshold:
    raise AssertionError()


if __name__ == '__main__':
  labelled_points = []
  detected_points = []

  labelled_points.append([0.5, 1.2, -2.3])
  detected_points.append([0.5, 1.5, -2.7])

  labelled_points.append([3.6, -1.8, 0.4])
  detected_points.append([3.5, -1.8, 0.4])

  labelled_points.append([-1.2, 2.5, 3.8])
  detected_points.append([-1.4, 2.4, 4.0])

  error_summary = error_analysis(np.array(labelled_points), np.array(detected_points))

  CheckEqual(error_summary.min_error, 0.1)
  CheckEqual(error_summary.max_error, 0.5)
  CheckEqual(error_summary.mean_error, 0.3)
  CheckEqual(error_summary.median_error, 0.3)
  CheckEqual(error_summary.l2_norm_error_list, [0.5, 0.1, 0.3])
  CheckEqual(error_summary.sorted_index_list, [0, 2, 1])