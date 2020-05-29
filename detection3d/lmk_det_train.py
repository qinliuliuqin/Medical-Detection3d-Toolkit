import argparse
import os

from detection3d.core.lmk_det_train import train

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    long_description = "Training engine for 3d medical image landmark detection"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='/home/ql/projects/Medical-Detection3d-Toolkit/detection3d/config/lmk_train_config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()

    train(args.input)


if __name__ == '__main__':
    main()
