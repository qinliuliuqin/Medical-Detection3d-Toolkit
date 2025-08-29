from easydict import EasyDict as edict
from detection3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer

__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

__C.general.training_image_list_file = '../assets/train.csv'

__C.general.validation_image_list_file = '../assets/val.csv'

# landmark label starts from 1, 0 represents the background.
__C.general.target_landmark_label = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
}

__C.general.save_dir = './saves/weights'

__C.general.resume_epoch = -1

__C.general.num_gpus = 1

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.crop_spacing = [2, 2, 2]      # mm

__C.dataset.crop_size = [96, 96, 96]   # voxel

__C.dataset.pad_size = [8, 8, 8]   # voxel, must be multiple of stride

__C.dataset.sampling_size = [6, 6, 6]      # voxel

__C.dataset.positive_upper_bound = 3    # voxel

__C.dataset.negative_lower_bound = 6    # voxel

__C.dataset.num_pos_patches_per_image = 8 # This should be same as number of landmarks 

__C.dataset.num_neg_patches_per_image = 16

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [AdaptiveNormalizer()]

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
__C.dataset.sampling_method = 'GLOBAL'

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

##################################
# data augmentation parameters
##################################

__C.augmentation = {}

__C.augmentation.turn_on = True

__C.augmentation.translation_lmk = True

# ------------------ Affine ------------------
__C.augmentation.affine_turn_on = True

__C.augmentation.scales = [0.9, 1.1]    # isotropic scale range

__C.augmentation.rotation = 10   # NOTE: despite the name, this is in degrees!

__C.augmentation.translation = 5  # mm

__C.augmentation.affine_p = 0.3 

# ------------------ Flip ------------------
__C.augmentation.flip_turn_on = True

__C.augmentation.flip_p = 0.5

# ------------------ Elastic deformation ------------------
__C.augmentation.elastic_turn_on = True

__C.augmentation.elastic_num_control_points = 4

__C.augmentation.elastic_max_displacement = 2.0   # mm

__C.augmentation.elastic_locked_borders = 1

__C.augmentation.elastic_p = 0.3

# ------------------ Motion ------------------
__C.augmentation.motion_turn_on = True

__C.augmentation.motion_num_transforms = 1

__C.augmentation.motion_p = 0.1

# ------------------ Noise ------------------
__C.augmentation.noise_turn_on = True

__C.augmentation.noise_mean = 0.0

__C.augmentation.noise_std = 0.02

__C.augmentation.noise_p = 0.3

# ------------------ Gamma ------------------
__C.augmentation.gamma_turn_on = True

__C.augmentation.log_gamma = [-0.2, 0.2]

__C.augmentation.gamma_p = 0.3

##################################
# loss function
##################################
__C.landmark_loss = {}

__C.landmark_loss.name = 'Focal'          # 'Dice', or 'Focal'

__C.landmark_loss.focal_obj_alpha = [0.75] * 9  # class balancing weight for focal loss

__C.landmark_loss.focal_gamma = 2         # gamma in pow(1-p,gamma) for focal loss

##################################
# net
##################################
__C.net = {}

__C.net.name = 'vdnet'

##################################
# training parameters
##################################
__C.train = {}

__C.train.use_amp = True

__C.train.epochs = 2001

__C.train.batch_size = 4

__C.train.num_threads = 4

__C.train.lr = 1e-4

__C.train.weight_decay = 1e-4

__C.train.betas = (0.9, 0.999)

__C.train.save_epochs = 1

##################################
# validation parameters
##################################
__C.val = {}

__C.val.interval = 1

__C.val.batch_size = 4

__C.val.num_threads = 4

__C.val.eval_fraction = 1

##################################
# debug parameters
##################################
__C.debug = {}

# random seed used in training
__C.debug.seed = 0

# whether to save input crops
__C.debug.save_inputs = False
