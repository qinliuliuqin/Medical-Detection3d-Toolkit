import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from detection3d.utils.image_tools import save_intermediate_results
from detection3d.loss.focal_loss import FocalLoss
from detection3d.utils.file_io import load_config, setup_logger, get_run_dir
from detection3d.utils.model_io import load_checkpoint, save_landmark_detection_checkpoint
from detection3d.dataset.dataloader import get_landmark_detection_dataloader
from tqdm import tqdm


def compute_landmark_mask_loss(outputs, landmark_masks, loss_function):
    """
    Computes the training loss for landmark mask segmentation, selecting only valid samples.

    Args:
        outputs (torch.Tensor): The output tensor from the model. Shape: [batch_size, channels, depth, height, width].
        landmark_masks (torch.Tensor): The ground truth landmark masks. Shape: [batch_size, channels, depth, height, width].
        loss_function (callable): The loss function to compute the loss.

    Returns:
        torch.Tensor: The computed training loss.
    """
    # Ensure outputs and landmark_masks have the same batch size
    assert outputs.shape[0] == landmark_masks.shape[0], "Outputs and landmark_masks batch sizes do not match"

    # Reshape outputs and landmark_masks for processing
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous()
    outputs = outputs.view(-1, outputs.shape[4])

    landmark_masks = landmark_masks.permute(0, 2, 3, 4, 1).contiguous()
    landmark_masks = landmark_masks.view(-1, landmark_masks.shape[4])

    # Select valid samples where the landmark mask is valid (>= 0)
    selected_sample_indices = torch.nonzero(landmark_masks[:, 0] >= 0, as_tuple=False).view(-1)

    landmark_masks = torch.index_select(landmark_masks, 0, selected_sample_indices)
    outputs = torch.index_select(outputs, 0, selected_sample_indices)

    # Compute the loss
    loss = loss_function(outputs, landmark_masks)

    return loss


def train_step(cfg, net, crops, landmark_masks, frames, filenames, loss_func, optimizer, scaler, batch_idx):
    net.train()

    if cfg.general.num_gpus > 0:
        crops = crops.cuda()
        landmark_masks = landmark_masks.cuda()

    optimizer.zero_grad()

    with autocast(device_type='cuda', enabled=cfg.train.use_amp):
        outputs = net(crops)
        train_loss = compute_landmark_mask_loss(outputs, landmark_masks, loss_func)

    # ---------- AMP path ----------
    if cfg.train.use_amp:
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # ---------- Standard FP32 path ----------
    else:
        train_loss.backward()
        optimizer.step()

    # Optional: Save debug outputs
    if cfg.debug.save_inputs:
        batch_size = crops.size(0)
        save_path = os.path.join(cfg.general.save_dir, f'batch_{batch_idx}')
        save_intermediate_results(
            list(range(batch_size)), crops, landmark_masks, outputs, frames, filenames, save_path
        )

    return train_loss


def val_step(cfg, network, val_data_loader, loss_function):
    """
    Perform a validation step over the entire validation dataset.

    Args:
        cfg: Configuration object containing relevant settings.
        network: The model/network to validate.
        val_data_loader: DataLoader for validation data.
        loss_function: Loss function to compute validation loss.

    Returns:
        avg_val_loss_per_sample: Average validation loss per sample.
    """
    network.eval()
    val_loss_epoch = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (crops, landmark_masks, landmark_coords, frames, filenames) in enumerate(val_data_loader):
            batch_size = crops.size(0)

            # Move to GPU if needed
            if cfg.general.num_gpus > 0:
                crops = crops.cuda()
                landmark_masks = landmark_masks.cuda()
                landmark_coords = landmark_coords.cuda()

            # Enable AMP if specified
            with autocast(device_type='cuda', enabled=cfg.train.use_amp):
                outputs = network(crops)
                val_loss = compute_landmark_mask_loss(outputs, landmark_masks, loss_function)

            val_loss_epoch += val_loss.item()
            total_samples += batch_size

            # Save for debugging/visualization
            if cfg.debug.save_inputs:
                save_path = os.path.join(cfg.general.save_dir, f'val_batch_{batch_idx}')
                save_intermediate_results(
                    list(range(batch_size)), crops, landmark_masks, outputs, frames, filenames, save_path
                )

    avg_val_loss_per_sample = val_loss_epoch / total_samples
    return avg_val_loss_per_sample


def log_training_setup(cfg, logger):
    logger.info("========== Training Configuration ==========")

    # --- Training parameters ---
    logger.info(">>> Training Parameters")
    logger.info(f"AMP enabled           : {cfg.train.use_amp}")
    logger.info(f"Resume from epoch     : {cfg.general.resume_epoch}")
    logger.info(f"Batch size            : {cfg.train.batch_size}")
    logger.info(f"Learning rate         : {cfg.train.lr}")
    logger.info(f"Crop size             : {cfg.dataset.crop_size}")
    logger.info(f"Loss function         : {cfg.landmark_loss.name}")

    # --- Validation parameters ---
    logger.info(">>> Validation Parameters")
    logger.info(f"Validation interval   : {cfg.val.interval} epochs")
    logger.info(f"Validation batch size : {cfg.val.batch_size}")
    logger.info(f"Validation threads    : {cfg.val.num_threads}")
    logger.info(f"Eval fraction         : {cfg.val.eval_fraction}")

    # --- Validation parameters ---
    logger.info(">>> Augmentation Status")
    logger.info(f"Augmentation turn on   : {cfg.augmentation.turn_on}")

    # --- Hardware & Save info ---
    logger.info(">>> Hardware / Runtime")
    if cfg.general.num_gpus > 0:
        logger.info(f"Using {cfg.general.num_gpus} GPU(s)")
        for i in range(cfg.general.num_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("Running on CPU")

    logger.info(f"Save directory        : {cfg.general.save_dir}")
    logger.info("============================================")


def train(config_file):
    """ Medical image segmentation training engine
    :param config_file: the absolute path of the input configuration file
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)
    mode = "current" if cfg.general.resume_epoch >= 0 else "next"
    cfg.general.save_dir =  get_run_dir(cfg.general.save_dir, mode)
    print(cfg.general.save_dir)

    scaler = GradScaler() if cfg.train.use_amp else None

    # control randomness during training
    np.random.seed(cfg.debug.seed)
    torch.manual_seed(cfg.debug.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.debug.seed)

    train_data_loader, num_modality, num_landmark_classes, num_train_cases = \
        get_landmark_detection_dataloader(cfg, 'train')

    val_data_loader, _, _, num_val_cases = \
        get_landmark_detection_dataloader(cfg, 'val')

    net_module = importlib.import_module('detection3d.network.' + cfg.net.name)
    net = net_module.Net(num_modality, num_landmark_classes)
    max_stride = net.max_stride()
    net_module.parameters_kaiming_init(net)

    # training optimizer
    opt = optim.AdamW(net.parameters(), lr=cfg.train.lr, betas=cfg.train.betas, weight_decay=cfg.train.weight_decay)

    if cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # load checkpoint if resume training
    if cfg.general.resume_epoch >= 0:
        start_epoch, start_batch_idx = load_checkpoint(cfg.general.resume_epoch, net, opt, scaler, cfg.general.save_dir)
    else:
        start_epoch, start_batch_idx = 0, 0

    if cfg.landmark_loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(
            class_num=num_landmark_classes, alpha=cfg.landmark_loss.focal_obj_alpha,
            gamma=cfg.landmark_loss.focal_gamma,use_gpu=cfg.general.num_gpus > 0)
        
    else:
        raise ValueError('Unknown loss function')
    
    # create a folder for saving training files if it does not exist
    os.makedirs(cfg.general.save_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(cfg.general.save_dir,f"tensorboard_e{start_epoch:04d}_b{start_batch_idx:06d}"))

    # copy the training config file
    shutil.copy(config_file, os.path.join(cfg.general.save_dir, os.path.basename(config_file)))

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'lmk_det3d')
    log_training_setup(cfg, logger)
    
    train_loader_iter = tqdm(
        enumerate(train_data_loader, start=start_batch_idx),
        total=len(train_data_loader),
        desc="Training",
        unit="batch"
    )

    batch_idx = start_batch_idx
    prev_epoch_idx = start_epoch
    train_loss_epoch = 0
    train_batch_size = 0

    print ("Training started with {} training cases and {} validation cases".format(num_train_cases, num_val_cases))

    for _, (crops, landmark_masks, landmark_coords, frames, filenames) in train_loader_iter:
        begin_t = time.time()
        batch_size = crops.size(0)
        train_batch_size += batch_size

        train_loss = train_step(cfg, net, crops, landmark_masks, frames, filenames, loss_func, opt, scaler, batch_idx)
        train_loss_epoch += train_loss.item()

        epoch_idx = batch_idx * cfg.train.batch_size // num_train_cases
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration / batch_size

        writer.add_scalar('Loss_batch/TrainLoss', train_loss.item(), batch_idx)

        # Validation and Logging
        if epoch_idx > prev_epoch_idx:
            avg_train_loss_per_sample = train_loss_epoch / train_batch_size

            force_val = (cfg.general.resume_epoch >= 0 and prev_epoch_idx == cfg.general.resume_epoch)
            if epoch_idx == 1 or epoch_idx % cfg.val.interval == 0 or force_val:
                avg_val_loss_per_sample = val_step(cfg, net, val_data_loader, loss_func)

            msg = 'epoch: {}, batch: {}, train_loss: {:.6f}, val_loss: {:.6f}, time: {:.4f} s/vol'
            logger.info(msg.format(epoch_idx, batch_idx, avg_train_loss_per_sample, avg_val_loss_per_sample, sample_duration))

            writer.add_scalars('Loss_epoch', {
                'Train': avg_train_loss_per_sample,
                'Validation': avg_val_loss_per_sample
                }, epoch_idx)

            # Reset per-epoch stats
            train_loss_epoch = 0
            train_batch_size = 0
            prev_epoch_idx = epoch_idx

            # Save checkpoint
            if epoch_idx % cfg.train.save_epochs == 0 and start_epoch != epoch_idx:
                save_landmark_detection_checkpoint(net, opt, scaler, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality)
                start_epoch = epoch_idx
        
        batch_idx += 1

    logger.info(f"Save directory        : {cfg.general.save_dir}")
    writer.close()