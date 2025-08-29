import glob
import importlib
import numpy as np
import torch.nn as nn
import os
import pandas as pd
import SimpleITK as sitk
import time
import torch
from easydict import EasyDict as edict
from detection3d.utils.file_io import load_config, get_resolved_run_dir
from detection3d.utils.model_io import get_checkpoint_folder
from detection3d.utils.image_tools import resample_spacing, pick_largest_connected_component, weighted_voxel_center, pad_image
from detection3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer
from detection3d.dataset.landmark_dataset import read_image_list
from typing import Dict, List
from monai.inferers import sliding_window_inference
import gc

def read_test_folder(folder_path):
    """ read single-modality input folder
    :param folder_path: image file folder path
    :return: a list of image path list, list of image case names
    """
    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    file = []
    for suf in suffix:
        file += glob.glob(os.path.join(folder_path, '*' + suf))

    file_name_list, file_path_list = [], []
    for im_pth in sorted(file):
        _, im_name = os.path.split(im_pth)
        for suf in suffix:
            idx = im_name.find(suf)
            if idx != -1:
                im_name = im_name[:idx]
                break
        file_name_list.append(im_name)
        file_path_list.append(im_pth)

    return file_name_list, file_path_list


def load_det_model(model_folder, gpu_id=0, chk_epoch=-1):
    """ load segmentation model from folder
    :param model_folder:    the folder containing the segmentation model
    :param gpu_id:          the gpu device id to run the segmentation model
    :return: a dictionary containing the model and inference parameters
    """
    assert os.path.isdir(model_folder), 'Model folder does not exist: {}'.format(
        model_folder)

    # load inference config file
    latest_checkpoint_dir, chk_epoch = get_checkpoint_folder(
        os.path.join(model_folder, 'checkpoints'), chk_epoch)
    
    infer_cfg = load_config(
        os.path.join(latest_checkpoint_dir, 'lmk_infer_config.py'))

    model = edict()
    model.infer_cfg = infer_cfg
    train_cfg = load_config(
        os.path.join(latest_checkpoint_dir, 'lmk_train_config.py'))
    model.train_cfg = train_cfg

    # load model state
    chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')

    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(int(gpu_id))
        # load network module
        state = torch.load(chk_file)
        net_module = importlib.import_module(
            'detection3d.network.' + state['net'])
        net = net_module.Net(state['in_channels'], state['num_landmark_classes'] + 1)
        net = nn.parallel.DataParallel(net, device_ids=[0])
        net.load_state_dict(state['state_dict'])
        net.eval()
        net = net.cuda()
        del os.environ['CUDA_VISIBLE_DEVICES']

    else:
        state = torch.load(chk_file, map_location='cpu')
        net_module = importlib.import_module(
            'detection3d.network.' + state['net'])
        net = net_module.Net(state['in_channels'], state['num_landmark_classes'] + 1)
        net.load_state_dict(state['state_dict'])
        net.eval()

    model.net = net
    model.crop_size, model.crop_spacing, model.max_stride, model.interpolation = \
        state['crop_size'], state['crop_spacing'], state['max_stride'], state['interpolation']
    model.in_channels, model.num_landmark_classes = \
        state['in_channels'], state['num_landmark_classes']
    model.pad_size = state['pad_size']

    model.crop_normalizers = []
    for crop_normalizer in state['crop_normalizers']:
        if crop_normalizer['type'] == 0:
            mean, stddev, clip = crop_normalizer['mean'], crop_normalizer['stddev'], \
                                crop_normalizer['clip']
            model.crop_normalizers.append(FixedNormalizer(mean, stddev, clip))

        elif crop_normalizer['type'] == 1:
            clip_sigma = crop_normalizer['clip_sigma']
            model.crop_normalizers.append(AdaptiveNormalizer(clip_sigma))

        else:
            raise ValueError('Unsupported normalization type.')

    return model, chk_epoch


def unwrap(model):
    """
    Unwrap a model from DataParallel or DistributedDataParallel if needed.
    Works recursively in case of nested wrapping.
    """
    # Drill down until no more .module attribute
    unwrapped = model
    while isinstance(unwrapped, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        unwrapped = unwrapped.module
    return unwrapped

def detect_voi_patches( model, iso_image: sitk.Image, use_gpu: bool = True, window_size: tuple[int, int, int] = (128, 128, 128),
                        batch_size: int = 1, overlap: float = 0.25, pad_mode: str = "constant", pad_value: float = 0.0,
                        amp_dtype = torch.float16):
    
    """
    Run sliding-window 3D inference on a SimpleITK image.

    Args:
        model: PyTorch model (optionally wrapped).
        iso_image: Preprocessed SimpleITK image.
        use_gpu: Use CUDA if available, else CPU.
        window_size: ROI size for sliding window.
        batch_size: Number of windows per forward pass.
        overlap: Overlap ratio between windows.
        pad_mode: Padding mode ("constant", "reflect", etc.).
        pad_value: Padding value if pad_mode="constant".
        amp_dtype: AMP precision (e.g. torch.float16) or None.

    Returns:
        np.ndarray: Logits array on CPU, shape [1, C, D, H, W].
    """
    # hygiene: free any leftovers before allocating
    torch.cuda.empty_cache()
    gc.collect()

    # unwrap and put model in eval/inference mode
    net = unwrap(model).eval()
    if use_gpu and torch.cuda.is_available():
        net = net.cuda()
    else:
        use_gpu = False  # force CPU path if no CUDA

    arr = sitk.GetArrayFromImage(iso_image)  # z,y,x
    input_tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

    if use_gpu:
        input_tensor = input_tensor.cuda(non_blocking=True)

    # sliding-window with GPU compute but CPU stitching ----
    sw_device = input_tensor.device if use_gpu else torch.device("cpu")
    out_device = torch.device("cpu")

    window_size = tuple(int(x) for x in window_size)

    with torch.inference_mode():
        # Automatic mixed precision reduces activation memory a lot
        autocast_ctx = (
            torch.amp.autocast(enabled=use_gpu and amp_dtype is not None, dtype=amp_dtype, device_type='cuda')
            if use_gpu else torch.autocast("cpu", enabled=False)
        )
        with autocast_ctx:
            logits = sliding_window_inference(
                inputs=input_tensor,                # [1,1,D,H,W]
                roi_size=window_size,                
                sw_batch_size=batch_size,           
                predictor=net,                      
                overlap=overlap,                    
                mode="gaussian",                    
                sigma_scale=0.125,
                padding_mode=pad_mode,
                cval=pad_value,
                sw_device=sw_device,                
                device=out_device,                  
            )

    # move result to CPU numpy and free GPU ----
    out_np = logits.float().cpu().numpy()
    del logits, input_tensor
    torch.cuda.empty_cache()
    gc.collect()
    return out_np


def image_preprocess(image, spacing=[0.3, 0.3, 0.3], max_stride=8, pad_size=None, interpolation='LINEAR', normalizer=None):
    """Resample, optional pad, and optional normalize an image.

    Args:
        image: SimpleITK image.
        spacing: Target spacing (xyz).
        max_stride: Grid alignment for resampling.
        pad_size: Optional pad size (xyz) to avoid edge effects.
        interpolation: Resampling method.
        normalizer: Optional callable to normalize image.

    Returns:
        SimpleITK.Image: Preprocessed image.
    """

    assert isinstance(image, sitk.Image)
    iso_image = resample_spacing(image, spacing, max_stride, interpolation)
    if pad_size:
        iso_image = pad_image(iso_image, pad_size)  # pad to avoid edge effects
    if normalizer is not None:
        iso_image = normalizer(iso_image)
    return iso_image


def postprocess_landmark_probs(landmark_mask_preds: np.ndarray, iso_image: sitk.Image,
                                landmark_name_to_label: Dict[str, int],prob_threshold: float = 0.8, 
                                background_label: int = 0, keep_largest_cc: bool = True) -> List[List[float]]:
    """Convert class probability maps into landmark detections.

    Args:
        landmark_mask_preds: Probabilities [C,D,H,W] or [1,C,D,H,W], channel 0 = background.
        iso_image: SimpleITK image carrying geometry.
        landmark_name_to_label: Map like {'Nasion': 1}.
        prob_threshold: Threshold before centroid.
        background_label: Channel index to skip.
        keep_largest_cc: Keep only largest 3D component.

    Returns:
        list[dict]: [{'lmk_name': str, 'x': float, 'y': float, 'z': float}, ...]
    """

    assert isinstance(iso_image, sitk.Image), "iso_image must be a SimpleITK Image"

    if landmark_mask_preds.ndim == 5:  # [N,C,D,H,W] -> [C,D,H,W]
        assert landmark_mask_preds.shape[0] == 1, "Expected batch size 1 in postprocess"
        landmark_mask_preds = landmark_mask_preds[0]

    if landmark_mask_preds.ndim != 4:
        raise ValueError(f"Expected probs of shape [C,D,H,W] or [1,C,D,H,W], got {landmark_mask_preds.shape}")

    # ensure float32
    if landmark_mask_preds.dtype != np.float32:
        landmark_mask_preds = landmark_mask_preds.astype(np.float32, copy=False)

    # sort landmarks by their numeric label (stable with your original code)
    landmark_names = list(landmark_name_to_label.keys())
    landmark_labels = np.array([landmark_name_to_label[n] for n in landmark_names], dtype=int)

    detected_landmarks: List[List[float]] = []
    landmark_prob_imgs = []

    for idx in np.argsort(landmark_labels):
        lmk_name = landmark_names[idx]
        lmk_channel = int(landmark_name_to_label[lmk_name])
        if lmk_channel == background_label:
            # skip background by definition
            continue

        # channel prob map
        landmark_mask_prob = landmark_mask_preds[lmk_channel]  # [D,H,W], float32

        # threshold → binary
        landmark_mask_binary = (landmark_mask_prob >= prob_threshold).astype(np.uint8)  # {0,1}
        landmark_mask_binary = sitk.GetImageFromArray(landmark_mask_binary)     # binary
        landmark_mask_binary.CopyInformation(iso_image)

        # pick the largest connected component by physical size
        if keep_largest_cc:
            landmark_mask_cc = pick_largest_connected_component(landmark_mask_binary, landmark_labels)

        # mask the probability by largest CC to compute max prob in component
        landmark_mask_cc = sitk.GetArrayFromImage(landmark_mask_cc).astype(np.float32)  # [D,H,W]

        masked_landmark_mask_prob = np.multiply(landmark_mask_cc, landmark_mask_prob)

        masked_landmark_mask_prob = sitk.GetImageFromArray(masked_landmark_mask_prob)
        masked_landmark_mask_prob.CopyInformation(iso_image)

        voxel_coordinate = weighted_voxel_center(masked_landmark_mask_prob, prob_threshold, 1.0)
        if voxel_coordinate is not None:
            # convert voxel coordinate to world coordinate (the voxel_coordinate should be in double-precision?)
            world_coordinate = masked_landmark_mask_prob.TransformContinuousIndexToPhysicalPoint(tuple(map(float, voxel_coordinate)))
            detected_landmarks.append({'lmk_name': lmk_name, 'x': float(world_coordinate[0]), 'y': float(world_coordinate[1]), 'z': float(world_coordinate[2])})
        else:
            detected_landmarks.append({'lmk_name': lmk_name, 'x': -1, 'y':  -1, 'z': -1})

    return detected_landmarks


def detection_single_image(image, model_folder, window_size=None, gpu_id=0, chk_epoch=-1,
                                batch_size=1, prob_threshold=0.8, overlap=0.25):
    
    """
    Run volumetric landmark detection on a single 3D image.

    Args:
        image: Input SimpleITK image.
        model_folder: Path to model or run_{n} folder.
        window_size: Sliding window size (default: 128³).
        gpu_id: GPU ID (default: 0).
        chk_epoch: Checkpoint epoch, -1 for latest.
        batch_size: Inference batch size.
        prob_threshold: Detection probability threshold.
        overlap: Sliding window overlap ratio.

    Returns:
        DataFrame of detected landmarks with coordinates.
    """

    assert isinstance(image, sitk.Image)
    # Load model
    run_dir = get_resolved_run_dir(model_folder)
    model, chk_epoch = load_det_model(run_dir, gpu_id, chk_epoch)

    # Load landmark label dictionary
    landmark_dict = model['train_cfg'].general.target_landmark_label

    if window_size is None:
        window_size = model['crop_size']
        print(f"No window size specified. Falling back to model's crop_size: {window_size}")

    assert np.all(np.array(window_size) % model['max_stride'] == 0), 'crop size not divisible by max stride'
    window_size = tuple(window_size)

    # Preprocess the image
    normalizer= model.crop_normalizers[0] if model['crop_normalizers'] is not None else None
    iso_image = image_preprocess(image, model['crop_spacing'], model['max_stride'],
                                    model['pad_size'], model['interpolation'],
                                    normalizer)

    landmark_mask_preds = detect_voi_patches(
        model.net,
        iso_image,
        use_gpu=(gpu_id is not None and gpu_id >= 0),
        window_size=window_size,  
        batch_size=batch_size,  
        overlap=overlap,                    
        pad_mode="constant",
        pad_value=0.0,
    )

        # Post-process
    detected_landmark = postprocess_landmark_probs(
            landmark_mask_preds=landmark_mask_preds,
            iso_image=iso_image,
            landmark_name_to_label=landmark_dict,
            prob_threshold=prob_threshold,
            background_label=0,
        )
    

    return pd.DataFrame(detected_landmark)

def detection(input, model_folder, gpu_id, return_landmark_file, save_landmark_file, save_prob,
                output_folder, window_size=None, over_lap =0.5, batch_size = 4, prob_threshold=0.5, chk_epoch=-1):
    
    """Run volumetric landmark detection on one or more images (patch-based).

    Args:
        input: Path to a volume, a directory, or a CSV list file.
        model_folder: Model directory or run_{n} folder.
        gpu_id: GPU id (>=0) or -1/None for CPU.
        return_landmark_file: If True, return detections.
        save_landmark_file: If True, write detections to disk.
        save_prob: If True, save per-class probability volumes.
        output_folder: Output directory.
        window_size: Sliding-window size, default model crop_size if None.
        over_lap: Overlap ratio for sliding-window inference.
        batch_size: Number of windows per forward pass.
        prob_threshold: Min probability to accept a landmark.
        chk_epoch: Checkpoint epoch (-1 = latest).

    Returns:
        pandas.DataFrame | None: Detections if requested, else None.
    """
    
    run_dir = get_resolved_run_dir(model_folder)
    # Load model
    begin = time.time()
    model, chk_epoch = load_det_model(run_dir, gpu_id, chk_epoch)
    load_model_time = time.time() - begin

    # Load landmark label dictionary
    landmark_dict = model['train_cfg'].general.target_landmark_label

    if window_size is None:
        window_size = model['crop_size']
        print(f"No window size specified. Falling back to model's crop_size: {window_size}")


    assert np.all(np.array(window_size) % model['max_stride'] == 0), 'crop size not divisible by max stride'
    window_size = tuple(window_size)

    # Load test images
    if os.path.isfile(input):
        if input.endswith('.csv'):
            file_name_list, file_path_list, _, _ = read_image_list(input, 'test')
        else:
            file_name_list = [os.path.basename(input)]
            file_path_list = [input]
    elif os.path.isdir(input):
        file_name_list, file_path_list = read_test_folder(input)
    else:
        raise ValueError(f"Unsupported input path: {input}")
    

    if save_landmark_file or save_prob:
        run_name = run_dir.split("/")[-1]
        output_folder = os.path.join(output_folder, f"{run_name}_epoch{chk_epoch}_results")
        os.makedirs(output_folder, exist_ok=True)


    # Collect results
    all_results = []
    for i, (file_path, file_name) in enumerate(zip(file_path_list, file_name_list)):
        file_name = file_name.removesuffix(".nii.gz")
        print(f"[{i}/{len(file_name_list)}] Processing: {file_name}")

        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load and resample image
        start_read = time.time()
        image = sitk.ReadImage(file_path, sitk.sitkFloat32)
        read_image_time = time.time() - start_read

        normalizer= model.crop_normalizers[0] if model['crop_normalizers'] is not None else None

        iso_image = image_preprocess(image, model['crop_spacing'], model['max_stride'], model['pad_size'],
                                      model['interpolation'], normalizer)


        # Forward pass
        start_infer = time.time()

        landmark_mask_preds = detect_voi_patches(
            model.net,
            iso_image,
            use_gpu=(gpu_id is not None and gpu_id >= 0),
            window_size=window_size, 
            batch_size=batch_size,  
            overlap=over_lap,                    
            pad_mode="constant",
            pad_value=0.0,
        )

        infer_time = time.time() - start_infer

        # Post-process
        start_post = time.time()
        detected_landmark = postprocess_landmark_probs(
            landmark_mask_preds=landmark_mask_preds,                   
            iso_image=iso_image,
            landmark_name_to_label=landmark_dict,
            prob_threshold=prob_threshold,
            background_label=0,
        )

        if return_landmark_file:
            tagged_landmarks = []
            for item in detected_landmark:
                tagged_item = item.copy()  # shallow copy
                tagged_item['image_id'] = file_name
                tagged_landmarks.append(tagged_item)
            all_results.extend(tagged_landmarks)


        if save_landmark_file or save_prob:
            detected_landmark_save_path = os.path.join(output_folder, file_name)
            os.makedirs(detected_landmark_save_path, exist_ok=True)

        if save_landmark_file:
            detected_landmark_df_path = os.path.join(detected_landmark_save_path,  '{}.csv'.format(file_name))
            detected_landmark_df = pd.DataFrame(detected_landmark)
            detected_landmark_df.to_csv(detected_landmark_df_path, index=False)

        if save_prob:
            # landmark_mask_preds shape: (num_classes+1, D, H, W), channel 0 = background
            if landmark_mask_preds.ndim == 5 and landmark_mask_preds.shape[0] == 1:
                landmark_mask_preds = np.squeeze(landmark_mask_preds, axis=0)

            for class_id in range(1, model['num_landmark_classes'] + 1):
                arr = landmark_mask_preds[class_id].astype(np.float32)   # (D, H, W)

                img = sitk.GetImageFromArray(arr)
                img.CopyInformation(iso_image)

                prob_path = os.path.join(
                    detected_landmark_save_path,
                    f"{file_name}_{class_id}.mha"
                )
                sitk.WriteImage(img, prob_path)

        post_time = time.time() - start_post
        print(f"⏱ read: {read_image_time+load_model_time:.2f}s  | prediction: {infer_time:.2f}s | saving: {post_time:.2f}s")
        
    if return_landmark_file:
        result_df = pd.DataFrame(all_results)
        cols = ["image_id"] + [c for c in result_df.columns if c != "image_id"]
        return result_df[cols]