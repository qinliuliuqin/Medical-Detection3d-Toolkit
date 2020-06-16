import copy
import os
import pandas as pd

from detection3d.vis.error_analysis import error_analysis
from detection3d.vis.gen_images import get_landmarks_stat


def add_document_text(original_text, new_text_to_add):
  """
  Add document text file
  """
  return original_text + r'+"{0}"'.format(new_text_to_add)


def write_summary_csv_report_for_all_landmarks(error_summary, csv_file_path):
  """
  Write a html report for all landmarks to summary the detection results
  """
  summary = []
  for landmark_idx, landmark_name in enumerate(error_summary.all_cases.keys()):
    num_pos_cases = len(error_summary.tp_cases[landmark_name]) + \
                    len(error_summary.fn_cases[landmark_name])
    num_neg_cases = len(error_summary.tn_cases[landmark_name]) + \
                    len(error_summary.fp_cases[landmark_name])
    if len(error_summary.tp_cases[landmark_name]) == 0 and num_pos_cases == 0:
      tpr = 100
    else:
      tpr = len(error_summary.tp_cases[landmark_name]) / max(1, num_pos_cases) * 100
    if len(error_summary.tn_cases[landmark_name]) == 0 and num_neg_cases == 0:
      tnr = 100
    else:
      tnr = len(error_summary.tn_cases[landmark_name]) / max(1, num_neg_cases) * 100
    mean_error = error_summary.mean_error_tp[landmark_name]
    std_error = error_summary.std_error_tp[landmark_name]
    median_error = error_summary.median_error_tp[landmark_name]
    max_error = error_summary.max_error_tp[landmark_name]
    summary.append([landmark_idx, landmark_name, num_pos_cases, num_neg_cases,
                    tpr, tnr, mean_error, std_error, median_error, max_error])
  
  columns = ['landmark_idx', 'landmark_name', 'pos_cases', 'neg_cases', 'TPR (%)', 'TNR (%)',
             'mean error (mm)', 'stddev', 'median error (mm)', 'max error (mm)']
  df = pd.DataFrame(data=summary, columns=columns)
  df.to_csv(csv_file_path, index=False, float_format='%.2f')


def write_html_report_for_single_landmark(document_text, analysis_text, html_report_path, width):
  """
  Write the html report for a landmark.
  """
  f = open(html_report_path, 'w')
  message = """
    <html>
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
      <title>result analysis</title>
      <style type="text/css">
          *{
              padding:0;
              margin:0;
          }
          .content {
              width: %spx;
              z-index:2;
          }
          .content img {
              width: %spx;
              transition-duration:0.2s;
              z-index:1;
          }
          .content img:active {
              transform: scale(2);
              -webkit-transform: scale(2); /*Safari 和 Chrome*/
              -moz-transform: scale(2); /*Firefox*/
              -ms-transform: scale(2); /*IE9*/
              -o-transform: scale(2); /*Opera*/
          }
      </style>
    </head>
    <body>
      <h1> Summary:</h1>
      %s
      <script type="text/javascript">
        document.write(%s)               
      </script>
    </body>
    </html>""" % (width, width, analysis_text, document_text)

  f.write(message)
  f.close()


def add_three_images(document_text, image_link_template, image_folder, images, width):
  """
  Add three plane images to the document text file
  """
  for idx in range(3):
    document_text += "\n"
    image_info = r'<td>{0}</td>'.format(image_link_template.format(
      os.path.join(image_folder, images[idx]), width))
    document_text = add_document_text(document_text, image_info)
  return document_text


def gen_html_report(landmarks_list, usage_flag, output_folder):
  """
  Generate landmark evaluation HTML report.
  """
  labelled_landmarks = landmarks_list[0]

  if usage_flag == 2:
    detected_landmarks = landmarks_list[1]
    assert len(labelled_landmarks.keys()) == len(detected_landmarks.keys())

    # sort the labelled landmarks according to detection error
    error_summary = error_analysis(labelled_landmarks, detected_landmarks)

  landmark_name_list = labelled_landmarks[list(labelled_landmarks.keys())[0]].keys()
  for landmark_idx, landmark_name in enumerate(landmark_name_list):
    print("Generating html report for landmark {}: {}.".format(landmark_idx, landmark_name))
    image_link_template = r"<div class='content'><img border=0  src= '{0}'  hspace=1  width={1} class='pic'></div>"
    error_info_template = r'<b>Labelled</b>: [{0:.2f}, {1:.2f}, {2:.2f}];'
    document_text = r'"<h1>check predicted coordinates:</h1>"'
    document_text += "\n"

    if usage_flag == 1:
      image_list = list(labelled_landmarks.keys())
      for image_idx, image_name in enumerate(image_list):
        label_landmark_world = labelled_landmarks[image_name][landmark_name]
        document_text = \
          gen_row_for_html(usage_flag, image_link_template, error_info_template,
                           document_text, image_list, image_idx, landmark_name,
                           [label_landmark_world], None)
        
    elif usage_flag == 2:
      image_list = error_summary.all_cases[landmark_name]
      error_sorted_index = error_summary.error_sorted_index
      for image_idx in error_sorted_index[landmark_name]:
        image_name = image_list[image_idx]
        label_landmark_world = labelled_landmarks[image_name][landmark_name]
        detected_landmark_world = detected_landmarks[image_name][landmark_name]
        error_info_template = r'<b>Labelled</b>: [{0:.2f}, {1:.2f}, {2:.2f}];'
        error_info_template += r'<b>Detected</b>: [{3:.2f}, {4:.2f}, {5:.2f}];  '
        error_info_template += r'<b>Type</b>: {6};'
        error_info_template += r'<b>Error</b>: x:{7:.2f}; y:{8:.2f}; z:{9:.2f}; L2:{10:.2f};'
        document_text = \
          gen_row_for_html(usage_flag, image_link_template, error_info_template,
                           document_text, image_list, image_idx, landmark_name,
                           [label_landmark_world, detected_landmark_world],
                            error_summary)

    else:
      raise ValueError('Undefined usage flag!')

    if usage_flag == 1:
      analysis_text = gen_analysis_text(len(image_list), usage_flag,
                                        labelled_landmarks, landmark_name, None)

    elif usage_flag == 2:
      analysis_text = gen_analysis_text(len(image_list), usage_flag,
                                        labelled_landmarks, landmark_name, error_summary)

    else:
      raise ValueError('Undefined usage float!')

    html_report_name = 'result_analysis.html'
    html_report_folder = os.path.join(output_folder, 'lm{}'.format(landmark_idx))
    if not os.path.isdir(html_report_folder):
      os.makedirs(html_report_folder)
    
    html_report_path = os.path.join(html_report_folder, html_report_name)
    write_html_report_for_single_landmark(document_text, analysis_text, html_report_path, width=200)

  if usage_flag == 2:
    summary_csv_report_name = 'summary.csv'
    summary_csv_path = os.path.join(output_folder, summary_csv_report_name)
    write_summary_csv_report_for_all_landmarks(error_summary, summary_csv_path)


def gen_analysis_text(num_data, usage_flag, labelled_landmark, landmark_name, error_summary):
  """
  Generate error analysis text for the html report.
  """
  analysis_text = r'<p style="color:red;">Basic information:</p>'
  analysis_text += '<p style="color:black;">Landmark name: {0}.</p>'.format(landmark_name)
  analysis_text += '<p style="color:black;"># cases in total: {0}.</p>'.format(num_data)
  labelled_landmarks_stat = get_landmarks_stat(labelled_landmark)
  
  analysis_text += r'<p style="color:black;"># cases having this landmark (Pos. cases): {0}.</p>'.format(
    len(labelled_landmarks_stat[landmark_name]['pos']))
  analysis_text += r'<p style="color:black;"># cases missing this landmark (Neg. cases): {}.</p>'.format(
    len(labelled_landmarks_stat[landmark_name]['neg']))
  if len(labelled_landmarks_stat[landmark_name]['neg']) > 0:
    missing_cases = copy.deepcopy(labelled_landmarks_stat[landmark_name]['neg'])
    missing_cases.sort()
    analysis_text += r'{}'.format(missing_cases)

  if usage_flag == 2:
    tp_cases = error_summary.tp_cases[landmark_name]
    tn_cases = error_summary.tn_cases[landmark_name]
    fp_cases = error_summary.fp_cases[landmark_name]
    fn_cases = error_summary.fn_cases[landmark_name]
    num_pos_cases = len(tp_cases) + len(fn_cases)
    num_neg_cases = len(tn_cases) + len(fp_cases)
    # compute TPR, TNR, FPR, FNR
    TPR = len(tp_cases) / max(1, num_pos_cases) * 100 \
      if len(tp_cases) != 0 or num_pos_cases != 0 else 100
    TNR = len(tn_cases) / max(1, num_neg_cases) * 100 \
      if len(tn_cases) != 0 or num_neg_cases != 0 else 100
    FPR = 100 - TNR
    FNR = 100 - TPR
    mean_error = error_summary.mean_error_tp[landmark_name]
    std_error = error_summary.std_error_tp[landmark_name]
    median_error = error_summary.median_error_tp[landmark_name]
    max_error = error_summary.max_error_tp[landmark_name]
    analysis_text += r'<p style="color:red;"> Landmark classification error: </p>'
    analysis_text += r'<p style="color:black;">TP (TPR): {0} ({1:.2f}%)</p>'.format(
      len(tp_cases), TPR)
    analysis_text += r'<p style="color:black;">TN (TNR): {0} ({1:.2f}%)</p>'.format(
      len(tn_cases), TNR)
    analysis_text += r'<p style="color:black;">FP (FPR): {0} ({1:.2f}%)</p>'.format(
      len(fp_cases), FPR)
    analysis_text += r'<p style="color:black;">FN (FNR): {0} ({1:.2f}%)</p>'.format(
      len(fn_cases), FNR)
    analysis_text += r'<p style="color:red;"> Landmark distance error for the {} TP cases (unit: mm): </p>'.format(
      len(tp_cases))
    analysis_text += r'<p style="color:black;">mean (std): {0:.2f} ({1:.2f})</p>'.format(
      mean_error, std_error)
    analysis_text += r'<p style="color:black;">median: {0:.2f}</p>'.format(median_error)
    analysis_text += r'<p style="color:black;">max: {0:.2f}</p>'.format(max_error)

  return analysis_text


def gen_row_for_html(usage_flag, image_link_template, error_info_template, document_text, image_list,
                 image_idx, landmark_name, landmark_worlds, error_summary, picture_folder='./pictures', width=200):
  """
  Generate a line of html text contents for labelled cases, in the usage of label checking.
  """
  image_name = image_list[image_idx]
  image_basename = image_name.split('/')[0]
  case_info = r'<b>Case nunmber</b>:{0} : {1} ,   '.format(image_idx, image_name)

  labelled_images = [image_basename + '_label_lm{}_axial.png'.format(landmark_name),
                     image_basename + '_label_lm{}_coronal.png'.format(landmark_name),
                     image_basename + '_label_lm{}_sagittal.png'.format(landmark_name)]
  labelled_point = landmark_worlds[0]
  
  if usage_flag == 1:
    error_info = error_info_template.format(landmark_worlds[0][0],
                                            landmark_worlds[0][1],
                                            landmark_worlds[0][2])
  
  elif usage_flag == 2:
    detected_images = [image_basename + '_detection_lm{}_axial.png'.format(landmark_name),
                       image_basename + '_detection_lm{}_coronal.png'.format(landmark_name),
                       image_basename + '_detection_lm{}_sagittal.png'.format(landmark_name)]
    detected_point = landmark_worlds[1]

    assert error_summary is not None
    x_error = error_summary.error_dx[landmark_name][image_idx]
    y_error = error_summary.error_dy[landmark_name][image_idx]
    z_error = error_summary.error_dz[landmark_name][image_idx]
    l2_error = error_summary.error_l2[landmark_name][image_idx]
    type_error = error_summary.error_type[landmark_name][image_idx]
    error_info = error_info_template.format(labelled_point[0],
                                            labelled_point[1],
                                            labelled_point[2],
                                            detected_point[0],
                                            detected_point[1],
                                            detected_point[2],
                                            type_error,
                                            x_error,
                                            y_error,
                                            z_error,
                                            l2_error)
  else:
    raise ValueError('Unsupported flag type!')

  document_text = add_document_text(document_text, case_info)
  document_text = add_document_text(document_text, error_info)
  
  document_text += "\n"
  document_text = add_document_text(document_text, "<table border=1><tr>")
  document_text = add_three_images(document_text, image_link_template, picture_folder, labelled_images, width)
  if usage_flag == 2:
    document_text = add_three_images(document_text, image_link_template, picture_folder, detected_images, width)
  document_text += "\n"
  document_text = add_document_text(document_text, r'</tr></table>')

  return document_text
