import pandas as pd


# get landmark names for each location
skull_landmark_names_file = '/home/ql/debug/skull.csv'
mandible_landmark_names_file = '/home/ql/debug/mandible.csv'
soft_tissue_landmark_names_file = '/home/ql/debug/soft_tissue.csv'

skull_df = pd.read_csv(skull_landmark_names_file)
mandible_df = pd.read_csv(mandible_landmark_names_file)
soft_tissue_df = pd.read_csv(soft_tissue_landmark_names_file)

skull_names_set = set(skull_df['landmark_name'])
mandible_names_set = set(mandible_df['landmark_name'])
soft_tissue_names_set = set(soft_tissue_df['landmark_name'])

assert len(skull_names_set) == 74 and len(mandible_names_set) == 66 and len(soft_tissue_names_set) == 41

# read detected landmark file
detected_landmark_file = '/home/ql/debug/detected_sample.csv'
detected_landmark_df = pd.read_csv(detected_landmark_file)

# save detected landmarks to txt files
skull_txt_path = '/home/ql/debug/skull.txt'
skull_txt = open(skull_txt_path, 'w')
for landmark_name in skull_names_set:
    content = detected_landmark_df[detected_landmark_df['name'] == landmark_name]
    if len(content) > 0:
        skull_txt.write('{}, {}, {}, {};'.format(landmark_name, content['x'].values[0], content['y'].values[0], content['z'].values[0]))
skull_txt.close()

mandible_txt_path = '/home/ql/debug/mandible.txt'
mandible_txt = open(mandible_txt_path, 'w')
for landmark_name in mandible_names_set:
    content = detected_landmark_df[detected_landmark_df['name'] == landmark_name]
    if len(content) > 0:
        mandible_txt.write('{}, {}, {}, {};'.format(landmark_name, content['x'].values[0], content['y'].values[0], content['z'].values[0]))
mandible_txt.close()

soft_tissue_txt_path = '/home/ql/debug/soft_tissue.txt'
soft_tissue_txt = open(soft_tissue_txt_path, 'w')
for landmark_name in soft_tissue_names_set:
    content = detected_landmark_df[detected_landmark_df['name'] == landmark_name]
    if len(content) > 0:
        soft_tissue_txt.write('{}, {}, {}, {};'.format(landmark_name, content['x'].values[0], content['y'].values[0], content['z'].values[0]))
soft_tissue_txt.close()

