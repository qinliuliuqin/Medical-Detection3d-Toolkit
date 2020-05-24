import argparse
import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from segmentation3d.utils.image_tools import save_intermediate_results
from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.loss.focal_loss import FocalLoss
from segmentation3d.utils.file_io import load_config, setup_logger
from segmentation3d.utils.model_io import load_checkpoint
from detection.dataloader.dataset import LandmarkDetectionDataset
from detection.utils.model_io import save_checkpoint


def train(config_file):
    """ Medical image segmentation training engine
    :param config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)

    # clean the existing folder if training from scratch
    if os.path.isdir(cfg.general.save_dir):
        if cfg.general.resume_epoch < 0:
            shutil.rmtree(cfg.general.save_dir)
            os.makedirs(cfg.general.save_dir)
            shutil.copy(config_file, os.path.join(cfg.general.save_dir, 'train_config.py'))
        else:
            shutil.copy(config_file, os.path.join(cfg.general.save_dir, 'train_config.py'))
    else:
        os.makedirs(cfg.general.save_dir)
        shutil.copy(config_file, os.path.join(cfg.general.save_dir, 'train_config.py'))

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'seg3d')

    # control randomness during training
    np.random.seed(cfg.debug.seed)
    torch.manual_seed(cfg.debug.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.debug.seed)

    # dataset
    dataset = LandmarkDetectionDataset(
      mode='train',
      image_list_file=cfg.general.training_image_list_file,
      target_landmark_label=cfg.general.target_landmark_label,
      target_organ_label=cfg.general.target_organ_label,
      crop_size=cfg.dataset.crop_size,
      crop_spacing=cfg.dataset.crop_spacing,
      sampling_method=cfg.dataset.sampling_method,
      sampling_size=cfg.dataset.sampling_size,
      positive_upper_bound=cfg.dataset.positive_upper_bound,
      negative_lower_bound=cfg.dataset.negative_lower_bound,
      num_pos_patches_per_image=cfg.dataset.num_pos_patches_per_image,
      num_neg_patches_per_image=cfg.dataset.num_neg_patches_per_image,
      augmentation_turn_on=cfg.augmentation.turn_on,
      augmentation_orientation_axis=cfg.augmentation.orientation_axis,
      augmentation_orientation_radian=cfg.augmentation.orientation_radian,
      augmentation_translation=cfg.augmentation.translation,
      interpolation=cfg.dataset.interpolation,
      crop_normalizers=cfg.dataset.crop_normalizers)

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batch_size,
                             num_workers=cfg.train.num_threads, pin_memory=True)

    net_module = importlib.import_module('detection.network.' + cfg.net.name)
    net = net_module.Net(dataset.num_modality(), dataset.num_landmark_classes)
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
          class_num=dataset.num_landmark_classes, alpha=cfg.landmark_loss.focal_obj_alpha,
          gamma=cfg.landmark_loss.focal_gamma,use_gpu=cfg.general.num_gpus > 0
        )
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(cfg.general.save_dir, 'tensorboard'))

    batch_idx = batch_start
    data_iter = iter(data_loader)

    # loop over batches
    for i in range(len(data_loader)):
        begin_t = time.time()

        crops, organ_masks, landmark_masks, landmark_coords, frames, filenames = data_iter.next()

        if cfg.general.num_gpus > 0:
            crops, organ_masks, landmark_masks, landmark_coords = \
              crops.cuda(), organ_masks.cuda(), landmark_masks.cuda(), landmark_coords.cuda()

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

        epoch_idx = batch_idx * cfg.train.batch_size // len(dataset)
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
                save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, dataset.num_modality())
                last_save_epoch = epoch_idx

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    long_description = "Training engine for 3d medical image landmark detection"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='./config/train_config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()

    train(args.input)


if __name__ == '__main__':
    main()
