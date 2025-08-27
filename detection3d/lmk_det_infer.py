import argparse

from detection3d.core.lmk_det_infer import detection


def main():
    long_description = 'Inference engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Single image\n' \
                       '2. A text file containing paths of all testing images\n' \
                       '3. A folder containing all testing images\n'
    
    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument("-i", "--input", default="../assets/case_001.nii.gz",
        help="Input folder/file for intensity images")
    
    parser.add_argument("-m", "--model", default="./saves/weights",
        help="Model root folder containing checkpoints")
    
    parser.add_argument("-o", "--output", default="./saves/results",
        help="Output folder for results")
    
    parser.add_argument("-g", "--gpu_id", type=int, default=0,
        help="GPU ID to run model, set to -1 for CPU only")
    
    parser.add_argument("-s", "--save_prob", action="store_true",
        help="Whether to save probability/heatmap maps")
    
    parser.add_argument("--return_landmark_file", action="store_true",
        help="If set, return detections instead of only saving them")
    
    parser.add_argument("--save_landmark_file", action="store_true", default=True,
        help="If set, save landmark detections to output folder")
    
    parser.add_argument("--window_size", type=int, nargs=3,
        help="Sliding-window size, e.g. --window_size 128 128 128 If not specified," \
        " inference will use the crop_spacing defined during model training.")
    
    parser.add_argument("--overlap", type=float, default=0.5,
        help="Overlap ratio for sliding window inference")
    
    parser.add_argument("--batch_size", type=int, default=4,
        help="Batch size for inference windows")
    
    parser.add_argument("--prob_threshold", type=float, default=0.5,
        help="Confidence threshold for landmark detection")
    
    parser.add_argument("--chk_epoch", type=int, default=-1,
        help="Checkpoint epoch to load (-1 = last)")
    
    args = parser.parse_args()
    
    detection(
        input=args.input,
        model_folder=args.model,
        gpu_id=args.gpu_id,
        return_landmark_file=args.return_landmark_file,
        save_landmark_file=args.save_landmark_file,
        save_prob=args.save_prob,
        output_folder=args.output,
        window_size=args.window_size,
        over_lap=args.overlap,
        batch_size=args.batch_size,
        prob_threshold=args.prob_threshold,
        chk_epoch=args.chk_epoch,
    )


if __name__ == '__main__':
    main()
