import numpy as np
import SimpleITK as sitk
import vtk


def weighted_voxel_center(image, threshold_min, threshold_max):
    """
    Get the weighted voxel center.
    :param image:
    :return:
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    image_npy[image_npy < threshold_min] = 0
    image_npy[image_npy > threshold_max] = 0
    weight_sum = np.sum(image_npy)
    if weight_sum <= 0:
        return None

    image_npy_x = np.zeros_like(image_npy)
    for i in range(image_npy.shape[0]):
        image_npy_x[i , :, :] = i

    image_npy_y = np.zeros_like(image_npy)
    for j in range(image_npy.shape[1]):
        image_npy_y[:, j, :] = j

    image_npy_z = np.zeros_like(image_npy)
    for k in range(image_npy.shape[2]):
        image_npy_z[:, :, k] = k

    weighted_center_x = np.sum(np.multiply(image_npy, image_npy_x)) / weight_sum
    weighted_center_y = np.sum(np.multiply(image_npy, image_npy_y)) / weight_sum
    weighted_center_z = np.sum(np.multiply(image_npy, image_npy_z)) / weight_sum
    weighted_center = [weighted_center_z, weighted_center_y, weighted_center_x]

    return weighted_center


def mask_to_mesh(image_path, stl_path, label):
    """
    Convert binary mask to STL file.
    :param image_path: The input mask path.
    :param stl_path: The output STL file
    :param label: The label to convert to binary mask
    :return:
    """
    # read the file
    if image_path.endswith('.nii.gz'):
        reader = vtk.vtkNIFTIImageReader()

    elif image_path.endswith('.mha') or image_path.endswith('.mhd'):
        reader = vtk.vtkMetaImageReader()

    else:
        raise ValueError('Unsupported image type.')

    reader.SetFileName(image_path)
    reader.Update()

    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, label)  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf.GetOutput())
    else:
        smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(30)
    smoother.NonManifoldSmoothingOn()
    # The positions can be translated and scaled such that they fit within a range of [-1, 1]
    # prior to the smoothing computation
    smoother.NormalizeCoordinatesOn()
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    # save the output
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(stl_path)
    writer.Write()