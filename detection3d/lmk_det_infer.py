import argparse

from detection3d.core.lmk_det_infer import detection


def main():
    long_description = 'Inference engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Single image\n' \
                       '2. A text file containing paths of all testing images\n' \
                       '3. A folder containing all testing images\n'
    
    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', default='../assets/case_001.nii.gz',
                        help='input folder/file for intensity images')
    parser.add_argument('-m', '--model', default='./saves/weights',
                        help='model root folder')
    parser.add_argument('-o', '--output', default='./saves/results',
                        help='output folder for segmentation')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='the gpu id to run model, set to -1 if using cpu only.')
    parser.add_argument('-s', '--save_prob', type=bool, default=False,
                        help='Whether save the probability maps.')

    args = parser.parse_args()
    detection(args.input, args.model, args.gpu_id, False, True, args.save_prob, args.output)


if __name__ == '__main__':
    main()
