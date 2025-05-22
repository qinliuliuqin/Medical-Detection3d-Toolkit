import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

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


def train_step(cfg, net, crops, landmark_masks, landmark_coords, frames, filenames, loss_func, optimizer, batch_idx):
    """
    Executes a single training step: forward pass, loss computation, backward pass, and optimizer step.

    Args:
        cfg: Configuration object containing general/debug settings.
        net: The model to train.
        crops: Input volume tensor [B, C, D, H, W].
        landmark_masks: Ground truth landmark masks.
        landmark_coords: Ground truth landmark coordinates (optional).
        frames: Frame identifiers for visualization.
        filenames: Input file names for visualization.
        loss_func: The loss function to use.
        optimizer: The optimizer instance.
        batch_idx: Index of the current batch (for logging/visualization).

    Returns:
        train_loss: Training loss value.
    """
    net.train()

    if cfg.general.num_gpus > 0:
        crops = crops.cuda()
        landmark_masks = landmark_masks.cuda()
        landmark_coords = landmark_coords.cuda()

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = net(crops)

    # Optional: Save debug inputs and outputs
    if cfg.debug.save_inputs:
        batch_size = crops.size(0)
        save_path = os.path.join(cfg.general.save_dir, f'batch_{batch_idx}')
        save_intermediate_results(list(range(batch_size)), crops, landmark_masks, outputs, frames, filenames, save_path)

    # Compute loss and backpropagate
    train_loss = compute_landmark_mask_loss(outputs, landmark_masks, loss_func)
    train_loss.backward()
    optimizer.step()

    return train_loss


def val_step(cfg, network, val_data_loader, num_val_cases, loss_function):
    """
    Perform a validation step over the entire validation dataset.

    Args:
        cfg: Configuration object containing relevant settings.
        network: The model/network to validate.
        val_data_loader: DataLoader for validation data.
        num_val_cases: Total number of validation cases.
        loss_function: Loss function to compute validation loss.

    Returns:
        val_loss_epoch: Total validation loss over the entire dataset.
    """
    network.eval()  # Set the network to evaluation mode
    val_loss_epoch = 0

    with torch.no_grad():
        for batch_idx, (crops, landmark_masks, landmark_coords, frames, filenames) in enumerate(val_data_loader):
            # Transfer data to GPU if available
            if cfg.general.num_gpus > 0:
                crops = crops.cuda()
                landmark_masks = landmark_masks.cuda()
                landmark_coords = landmark_coords.cuda()

            # Run inference
            outputs = network(crops)

            # Save inputs and outputs for debugging/visualization
            if cfg.debug.save_inputs:
                batch_size = crops.size(0)
                save_path = os.path.join(cfg.general.save_dir, f'val_batch_{batch_idx}')
                save_intermediate_results(
                    list(range(batch_size)), crops, landmark_masks, outputs, frames, filenames, save_path
                )

            # Accumulate validation loss
            val_loss_epoch += compute_landmark_mask_loss(outputs, landmark_masks, loss_function)

    return (val_loss_epoch / num_val_cases).item()


def train(config_file):
    """ Medical image segmentation training engine
    :param config_file: the absolute path of the input configuration file
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)

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

    # loop over batches
    for i, (crops, landmark_masks, landmark_coords, frames, filenames) in enumerate(train_data_loader):
        begin_t = time.time()
        
        train_loss = train_step(cfg, net, crops, landmark_masks, landmark_coords, frames, filenames, loss_func, opt, batch_idx)

        epoch_idx = batch_idx * cfg.train.batch_size // num_train_cases
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batch_size

        if epoch_idx % cfg.val.interval == 0:
            val_loss_epoch = val_step(cfg, net, val_data_loader, num_val_cases, loss_func)

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, val_loss_epoch: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), val_loss_epoch, sample_duration)
        logger.info(msg)

        # save checkpoint
        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                save_landmark_detection_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality)
                last_save_epoch = epoch_idx


        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()