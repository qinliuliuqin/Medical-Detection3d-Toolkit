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

    # create save folder if it does not exist
    if not os.path.isdir(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)

    # Update training config file
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
          gamma=cfg.landmark_loss.focal_gamma,use_gpu=cfg.general.num_gpus > 0
        )
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(cfg.general.save_dir, 'tensorboard'))

    batch_idx = batch_start
    data_iter = iter(train_data_loader)

    # loop over batches
    for i in range(len(train_data_loader)):
        begin_t = time.time()

        crops, landmark_masks, landmark_coords, frames, filenames = data_iter.next()

        if cfg.general.num_gpus > 0:
            crops, landmark_masks, landmark_coords = \
              crops.cuda(), landmark_masks.cuda(), landmark_coords.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward and backward
        outputs = net(crops)

        # save training crops for visualization
        if cfg.debug.save_inputs:
            batch_size = crops.size(0)
            save_intermediate_results(list(range(batch_size)), crops, landmark_masks, outputs, frames, filenames,
                                      os.path.join(cfg.general.save_dir, 'batch_{}'.format(i)))

        # select valid samples for landmark mask segmentation
        assert outputs.shape[0] == landmark_masks.shape[0]
        outputs = outputs.permute(0, 2, 3, 4, 1).contiguous()
        outputs = outputs.view(-1, outputs.shape[4])
        landmark_masks = landmark_masks.permute(0, 2, 3, 4, 1).contiguous()
        landmark_masks = landmark_masks.view(-1, landmark_masks.shape[4])

        selected_sample_indices = torch.nonzero(landmark_masks[:, 0] >= 0).squeeze()
        landmark_masks = torch.index_select(landmark_masks, 0, selected_sample_indices)
        outputs = torch.index_select(outputs, 0, selected_sample_indices)

        train_loss = loss_func(outputs, landmark_masks)
        train_loss.backward()

        # update weights
        opt.step()

        epoch_idx = batch_idx * cfg.train.batch_size // num_train_cases
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batch_size

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)

        # save checkpoint
        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                save_landmark_detection_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality)
                last_save_epoch = epoch_idx

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()