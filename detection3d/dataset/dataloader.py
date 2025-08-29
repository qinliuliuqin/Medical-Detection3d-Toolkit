from torch.utils.data import DataLoader
from detection3d.dataset.landmark_dataset import LandmarkDetectionDataset
from detection3d.dataset.sampler import EpochConcateSampler, RandomSubsetSampler


def get_landmark_detection_dataloader(cfg, mode, seed=100):
    """
    Get landmark detection data loader
    """
    if mode == 'train':
        dataset = LandmarkDetectionDataset(
            mode='train',
            image_list_file=cfg.general.training_image_list_file,
            target_landmark_label=cfg.general.target_landmark_label,
            crop_size=cfg.dataset.crop_size,
            pad_size=cfg.dataset.pad_size,
            crop_spacing=cfg.dataset.crop_spacing,
            sampling_method=cfg.dataset.sampling_method,
            sampling_size=cfg.dataset.sampling_size,
            positive_upper_bound=cfg.dataset.positive_upper_bound,
            negative_lower_bound=cfg.dataset.negative_lower_bound,
            num_pos_patches_per_image=cfg.dataset.num_pos_patches_per_image,
            num_neg_patches_per_image=cfg.dataset.num_neg_patches_per_image,
            augmentation_turn_on=cfg.augmentation.turn_on,
            aug_cfg= cfg.augmentation,
            interpolation=cfg.dataset.interpolation,
            crop_normalizers=cfg.dataset.crop_normalizers)
        
        num_cases = len(dataset)
        sampler = EpochConcateSampler(dataset, cfg.train.epochs)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batch_size,
                                    num_workers=cfg.train.num_threads, pin_memory=True,
                                    persistent_workers=True)

    else:
        dataset = LandmarkDetectionDataset(
            mode='val',
            image_list_file=cfg.general.validation_image_list_file,
            target_landmark_label=cfg.general.target_landmark_label,
            crop_size=cfg.dataset.crop_size,
            pad_size=cfg.dataset.pad_size,
            crop_spacing=cfg.dataset.crop_spacing,
            sampling_method=cfg.dataset.sampling_method,
            sampling_size=cfg.dataset.sampling_size,
            positive_upper_bound=cfg.dataset.positive_upper_bound,
            negative_lower_bound=cfg.dataset.negative_lower_bound,
            num_pos_patches_per_image=cfg.dataset.num_pos_patches_per_image,
            num_neg_patches_per_image=cfg.dataset.num_neg_patches_per_image,
            augmentation_turn_on=cfg.augmentation.turn_on,
            aug_cfg= cfg.augmentation,
            interpolation=cfg.dataset.interpolation,
            crop_normalizers=cfg.dataset.crop_normalizers)

        num_cases = int(len(dataset) * cfg.val.eval_fraction)// 2 * 2
        sampler = RandomSubsetSampler(dataset, num_cases, seed= seed)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.val.batch_size,
                                    num_workers=cfg.val.num_threads, pin_memory=True, persistent_workers=True)

    return data_loader, dataset.num_modality(), dataset.num_landmark_classes, num_cases