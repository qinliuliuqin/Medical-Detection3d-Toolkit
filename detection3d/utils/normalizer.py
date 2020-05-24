import SimpleITK as sitk

from detection3d.utils.image_tools import normalize_image, get_mean_std_from_image


class FixedNormalizer(object):
  """
  Use fixed mean and stddev to normalize image intensities
  intensity = (intensity - mean) / stddev
  if clip is enabled:
      intensity = np.clip((intensity - mean) / stddev, -1, 1)
  """

  def __init__(self, mean, stddev, clip=True):
    """ constructor """
    assert stddev > 0, 'stddev must be positive'
    assert isinstance(clip, bool), 'clip must be a boolean'
    self.mean = mean
    self.stddev = stddev
    self.clip = clip

  def __call__(self, image):
    """ normalize image """
    if isinstance(image, sitk.Image):
      return normalize_image(image, self.mean, self.stddev, self.clip)

    elif isinstance(image, (list, tuple)):
      for idx, im in enumerate(image):
        assert isinstance(im, sitk.Image)
        image[idx] = normalize_image(im, self.mean, self.stddev, self.clip)
      return image

    else:
      raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

  def to_dict(self):
    """ convert parameters to dictionary """
    obj = {'type': 0, 'mean': self.mean, 'stddev': self.stddev, 'clip': self.clip}
    return obj


class AdaptiveNormalizer(object):
  """
  Normalize image using z-score normalization.
  """

  def __init__(self, clip_sigma=3):
    """
    :param clip_sigma: clip the intensity within the 'clip_sigma' standard deviation. 68% voxels lies within 1
      standard deviation, 95% within 2 standard deviation, and 99.7% within 3 standard deviation.
    """
    assert clip_sigma > 0
    self.clip_sigma = clip_sigma

  def normalize(self, single_image):
    """ Normalize a given image """
    assert isinstance(single_image, sitk.Image), 'image must be an image3d object'

    normalize_mean, normalize_stddev = get_mean_std_from_image(single_image)
    normalize_stddev = max(normalize_stddev, 1e-6)

    return normalize_image(single_image, normalize_mean, normalize_stddev, True, -self.clip_sigma, self.clip_sigma)

  def __call__(self, image):
    """ normalize image """
    if isinstance(image, sitk.Image):
      return self.normalize(image)

    elif isinstance(image, (list, tuple)):
      for idx, im in enumerate(image):
        assert isinstance(im, sitk.Image)
        image[idx] = self.normalize(im)
      return image

    else:
      raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

  def to_dict(self):
    """ convert parameters to dictionary """
    obj = {'type': 1, 'clip_sigma': self.clip_sigma}
    return obj