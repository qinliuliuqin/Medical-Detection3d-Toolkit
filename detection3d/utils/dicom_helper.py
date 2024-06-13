import os
import sys
import SimpleITK as sitk
import time


def read_dicom_series(data_directory):
  """
  Read Dicom series from disk
  :param data_directory:  the folder containing all dicom series
  :return image: the 3D image volume
  """
  # Read the original series. First obtain the series file names using the image series reader.
  series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
  if not series_IDs:
    print("ERROR: given directory \"" + data_directory + "\" does not contain a DICOM series.")
    sys.exit(1)
  series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])

  series_reader = sitk.ImageSeriesReader()
  series_reader.SetFileNames(series_file_names)

  # Configure the reader to load all of the DICOM tags (public + private):
  # By default tags are not loaded (saves time). If tags are loaded, the private tags are not loaded.
  # We explicitly configure the reader to load tags, including the private ones.
  series_reader.MetaDataDictionaryArrayUpdateOn()
  series_reader.LoadPrivateTagsOn()
  image = series_reader.Execute()

  return image


def write_dicom_series(image, directory, dtype=sitk.sitkInt16, tags=None):
  """
  Write dicom series to disk
  :param image: the input image volume in SimpleITK Image type
  :param directory: the directory to save the dicom series
  :param dtype: the output image type
  :param tags: the tags shared by all dicom series
  """
  assert isinstance(image, sitk.Image)

  writer = sitk.ImageFileWriter()
  writer.KeepOriginalImageUIDOn()

  if not os.path.isdir(directory):
    os.makedirs(directory)

  for i in range(image.GetDepth()):
    image_slice = image[:, :, i]

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(dtype)
    image_slice = castFilter.Execute(image_slice)

    # Tags shared by the series.
    if tags is not None:
      for key in tags.keys():
        image_slice.SetMetaData(key, tags[key])

    direction = image.GetDirection()
    internal_tags = {"0008|0031": time.strftime("%H%M%S"), # modification time,
                     "0008|0021": time.strftime("%Y%m%d"), # modification date
                     "0020|000e": "0.0.000.0.0.0000000.0.0000.0.00000000.00000000",
                     "0020|0037": "\\".join(map(str, (direction[0], direction[3], direction[6],
                                                      direction[1], direction[4], direction[7]))),
                     "0020|0032": '\\'.join(map(str, image.TransformIndexToPhysicalPoint((0, 0, i)))),
                     "0018|0050": str(image.GetSpacing()[2]),
                     "0020|0013": str(i)
                     }

    for key in internal_tags.keys():
      image_slice.SetMetaData(key, internal_tags[key])

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(directory, str(i) + '.dcm'))
    writer.Execute(image_slice)


def write_binary_dicom_series(image, directory, in_label=1, out_label=100, dtype=sitk.sitkInt16, tags=None):
  """
  Write dicom series to disk, each dicom series is a binary mask
  :param image: the input image volume in SimpleITK Image type
  :param directory: the directory to save the dicom series
  :param in_label: the input label, type: int
  :param out_label: the output label, type: int
  :param dtype: the output image type
  :param tags: the tags shared by all dicom series
  """
  assert isinstance(image, sitk.Image)

  # get the binary mask
  binary_image_npy = sitk.GetArrayFromImage(image)
  binary_image_npy[binary_image_npy != in_label] = 0
  binary_image_npy[binary_image_npy == in_label] = out_label
  binary_image = sitk.GetImageFromArray(binary_image_npy)
  binary_image.CopyInformation(image)

  writer = sitk.ImageFileWriter()
  writer.KeepOriginalImageUIDOn()

  if not os.path.isdir(directory):
    os.makedirs(directory)

  for i in range(binary_image.GetDepth()):
    image_slice = binary_image[:, :, i]

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(dtype)
    image_slice = castFilter.Execute(image_slice)

    # Tags shared by the series.
    if tags is None:
      tags = dicom_tags_dict()

    for key in tags.keys():
      image_slice.SetMetaData(key, tags[key])

    direction = binary_image.GetDirection()
    internal_tags = {"0008|0031": time.strftime("%H%M%S"), # modification time,
                     "0008|0021": time.strftime("%Y%m%d"), # modification date
                     "0020|000e": "0.0.000.0.0.0000000.0.0000.0.00000000.00000000",
                     "0020|0037": "\\".join(map(str, (direction[0], direction[3], direction[6],
                                                      direction[1], direction[4], direction[7]))),
                     "0020|0032": '\\'.join(map(str, binary_image.TransformIndexToPhysicalPoint((0, 0, i)))),
                     "0018|0050": str(binary_image.GetSpacing()[2]),
                     "0020|0013": str(i)
                     }

    for key in internal_tags.keys():
      image_slice.SetMetaData(key, internal_tags[key])

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(directory, str(i) + '.dcm'))
    writer.Execute(image_slice)


def dicom_tags_dict(modality='CT',
                    image_type=r'DERIVED\\SECONDARY',
                    conversion_type='DV',
                    patient_position='HFS',
                    series_description='IDEA',
                    study_description='DentalSeg',
                    patient_name='Anonymous',
                    patient_id='20200121',
                    patient_age='99',
                    rescale_type='HU',
                    rescale_slope='1',
                    rescale_intercept='0'):
  """
  Get basic dicom tags.
  """

  tags = dict()
  tags['0008|0060'] = modality
  tags['0008|0008'] = image_type
  tags['0008|0064'] = conversion_type
  tags['0008|103E'] = series_description
  tags['0008|1030'] = study_description
  tags['0018|5100'] = patient_position
  tags['0010|0010'] = patient_name
  tags['0010|0020'] = patient_id
  tags['0010|1010'] = patient_age
  tags['0028|1052'] = rescale_intercept
  tags['0028|1053'] = rescale_slope
  tags['0028|1054'] = rescale_type

  return tags