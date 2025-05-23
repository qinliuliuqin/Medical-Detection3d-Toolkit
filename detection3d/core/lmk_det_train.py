import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.amp import autocast, GradScaler
from detection3d.utils.image_tools import save_intermediate_results
from detection3d.loss.focal_loss import FocalLoss
from detection3d.utils.file_io import load_config, setup_logger
from detection3d.utils.model_io import load_checkpoint, save_landmark_detection_checkpoint
from detection3d.dataset.dataloader import get_landmark_detection_dataloader


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


def train_step(cfg, net, crops, landmark_masks, landmark_coords, frames, filenames, loss_func, optimizer, scaler, batch_idx):
    net.train()

    if cfg.general.num_gpus > 0:
        crops = crops.cuda()
        landmark_masks = landmark_masks.cuda()
        landmark_coords = landmark_coords.cuda()

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


def train(config_file):
    """ Medical image segmentation training engine
    :param config_file: the absolute path of the input configuration file
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)

    scaler = GradScaler() if cfg.train.use_amp else None


    # clean the existing folder if training from scratch
    if os.path.isdir(cfg.general.save_dir) and cfg.general.resume_epoch < 0:
        shutil.rmtree(cfg.general.save_dir)

    # create a folder for saving training files if it does not exist
    if not os.path.isdir(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)

    # update the training config file
    shutil.copy(config_file, os.path.join(cfg.general.save_dir, os.path.basename(config_file)))

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'lmk_det3d')

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

    if cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)

    # load checkpoint if resume epoch > 0
    if cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(cfg.general.resume_epoch, net, opt, cfg.general.save_dir)
    else:
        last_save_epoch, batch_start = 0, 0

    if cfg.landmark_loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(
            class_num=num_landmark_classes, alpha=cfg.landmark_loss.focal_obj_alpha,
            gamma=cfg.landmark_loss.focal_gamma,use_gpu=cfg.general.num_gpus > 0)
        
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(cfg.general.save_dir, 'tensorboard'))


    batch_idx = batch_start
    train_loss_epoch = 0
    train_batch_size = 0
    prev_epoch_idx = 0
    last_save_epoch = -1

    for i, (crops, landmark_masks, landmark_coords, frames, filenames) in enumerate(train_data_loader, start=batch_start):
        begin_t = time.time()
        batch_size = crops.size(0)
        train_batch_size += batch_size

        train_loss = train_step(cfg, net, crops, landmark_masks, landmark_coords, frames, filenames, loss_func, opt, scaler, batch_idx)
        train_loss_epoch += train_loss.item()

        epoch_idx = batch_idx * cfg.train.batch_size // num_train_cases
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration / batch_size

        # Validation and Logging
        if epoch_idx > prev_epoch_idx:
            avg_train_loss_per_sample = train_loss_epoch / train_batch_size
            val_loss_epoch = val_step(cfg, net, val_data_loader, loss_func)

            msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, val_loss: {:.4f}, time: {:.4f} s/vol'
            logger.info(msg.format(epoch_idx, batch_idx, avg_train_loss_per_sample, val_loss_epoch, sample_duration))

            writer.add_scalar('Val/Loss', val_loss_epoch, batch_idx)

            # Reset per-epoch stats
            train_loss_epoch = 0
            train_batch_size = 0
            prev_epoch_idx = epoch_idx

            # Save checkpoint
            if epoch_idx % cfg.train.save_epochs == 0 and last_save_epoch != epoch_idx:
                save_landmark_detection_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality)
                last_save_epoch = epoch_idx

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)
        batch_idx += 1

    writer.close()