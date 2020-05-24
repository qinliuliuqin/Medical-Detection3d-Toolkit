import glob
import os
import torch
import shutil


def get_checkpoint_folder(chk_root, epoch):
  """
  Get the checkpoint's folder with the specified epoch.
  :param chk_root: the check point root directory, which may contain multiple checkpoints.
  :param epoch: the epoch of the checkpoint, set -1 to get the latest epoch.
  :return: the folder containing the checkpoint with the specified epoch.
  """
  assert os.path.isdir(chk_root), 'The folder does not exist: {}'.format(chk_root)

  if epoch < 0:
    latest_epoch = -1
    chk_folders = os.path.join(chk_root, 'chk_*')
    for folder in glob.glob(chk_folders):
      folder_name = os.path.basename(folder)
      tokens = folder_name.split('_')
      cur_epoch = int(tokens[-1])
      if cur_epoch > latest_epoch:
        latest_epoch = cur_epoch

    epoch = latest_epoch

  return os.path.join(chk_root, 'chk_{}'.format(epoch))


def save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality):
    """ save model and parameters into a checkpoint file (.pth)

    :param net: the network object
    :param opt: the optimizer object
    :param epoch_idx: the epoch index
    :param batch_idx: the batch index
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param max_stride: the maximum stride of network
    :param num_modality: the number of image modalities
    :return: None
    """
    chk_folder = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)

    filename = os.path.join(chk_folder, 'params.pth')
    opt_filename = os.path.join(chk_folder, 'optimizer.pth')

    state = {'epoch':                 epoch_idx,
             'batch':                 batch_idx,
             'net':                   cfg.net.name,
             'max_stride':            max_stride,
             'state_dict':            net.state_dict(),
             'crop_spacing':          cfg.dataset.crop_spacing,
             'crop_size':             cfg.dataset.crop_size,
             'interpolation':         cfg.dataset.interpolation,
             'in_channels':           num_modality,
             'num_organ_classes':     len(cfg.general.target_organ_label),
             'num_landmark_classes': len(cfg.general.target_landmark_label),
             'crop_normalizers':      [normalizer.to_dict() for normalizer in cfg.dataset.crop_normalizers]}

    # save python check point
    torch.save(state, filename)

    # save python optimizer state
    torch.save(opt.state_dict(), opt_filename)

    # save training and inference configuration files
    config_folder = os.path.dirname(os.path.dirname(__file__))
    infer_config_file = os.path.join(os.path.join(config_folder, 'config', 'infer_config.py'))
    shutil.copy(infer_config_file, os.path.join(chk_folder, 'infer_config.py'))

    shutil.copy(os.path.join(cfg.general.save_dir, 'train_config.py'), os.path.join(chk_folder, 'train_config.py'))