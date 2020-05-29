import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['image.origin'] = 'lower'


"""
Plot horizontal and vertical crossing lines passing through the specified point on
 the image.
Input arguments:
  point_x: X coordinate of the point.
  point_t: Y coordinate of the point. 
  width:   The image width.
  height:  The image height.
  line_width: The width of the line in pixels.
  line_style: The style of the line.
"""
def PlotCrossing(point_x, point_y, width, height, line_style = 'r-.', line_width = 1):
    xx1 = np.arange(0, width)
    yy1 = np.ones(width) * point_y

    xx2 = np.ones(height) * point_x
    yy2 = np.arange(0, height)

    plt.plot(xx1, yy1, line_style, linewidth = line_width)
    plt.plot(xx2, yy2, line_style, linewidth = line_width)


def crop_plane(plane, center_x, center_y, size_x, size_y):
  """
  Crop a smaller patch from a plane.
  Input arguments:
  plane:    A 2D array, the input plane image. Shape: [dim_y, dim_x]
  center_x: X coordinate of the cropping center.
  center_y: Y coordinate of the cropping center.
  size_x:   The cropping width.
  size_y:   The cropping height.
  """
  assert isinstance(plane, np.ndarray)

  min_plane_val = np.min(plane)
  cropped_plane = np.zeros((size_y, size_x))
  cropped_plane.fill(min_plane_val)
  
  start_x = int(center_x - (size_x // 2))
  end_x = start_x + size_x

  start_y = int(center_y - (size_y // 2))
  end_y = start_y + size_y

  height, width = plane.shape
  valid_start_x, valid_start_y = max(0, start_x), max(0, start_y)
  valid_end_x, valid_end_y = min(width, end_x), min(height, end_y)
  
  cropped_start_x, cropped_end_x = valid_start_x - start_x, size_x + (valid_end_x - end_x)
  cropped_start_y, cropped_end_y = valid_start_y - start_y, size_y + (valid_end_y - end_y)
  cropped_plane[cropped_start_y: cropped_end_y, cropped_start_x:cropped_end_x] = \
    plane[valid_start_y:valid_end_y, valid_start_x:valid_end_x]

  return cropped_plane


"""
Plot crossing lines on a 2D plane image and save it to the specified path.
Input arguments:
  plane:      2-D plane image.
  point:      2-D dimensional point on the image space.
  image_path: The path to save the image.
  title_color: A string, color of the image's title.
  line_width: The width of the line in pixels.
  line_style: The style of the line.
"""
def PlotCrossingAndSavePlanes(plane, point, title, image_path, title_color='g',
                              line_style='g-.', line_width=1,
                              crop_x=100, crop_y=100):

  # Create a new figure
  fig = plt.figure(1, figsize=(5, 5))

  cropped_plane = crop_plane(np.transpose(plane), point[0], point[1], crop_x, crop_y)
  
  plt.imshow(cropped_plane, cmap='gray')
  cropped_height, cropped_width = cropped_plane.shape
  PlotCrossing(crop_x//2, crop_y//2, cropped_width, cropped_height, line_style, line_width)

  plt.axis('off')
  plt.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
  plt.title(title, fontsize=20, color=title_color)

  # Save and close the figure.
  fig.savefig(image_path)
  fig.clf()


def save_black_planes(title, image_path, title_color='g', crop_x=100, crop_y=100):
  """
  Save black planes for files that do not contain the corresponding landmark.
  Input arguments:
    title:       A string, title of the image.
    image_path:  A string ,path for the image to save.
    title_color: A string, color of the image's title.
    crop_x:      A number default to 100, the returned cropping width of the image.
    crop_y:      A number default to 100, the returned cropping height of the image.
  """
  fig = plt.figure(1, figsize=(5,5))
  plane = np.zeros([crop_x, crop_y])
  plt.imshow(plane, cmap='gray')
  plt.axis('off')

  plt.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
  plt.title(title, fontsize=20, color=title_color)

  # Save and close the figure.
  fig.savefig(image_path)
  fig.clf()