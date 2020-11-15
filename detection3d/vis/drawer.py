## numpy is used for creating fake data
import numpy as np
# import matplotlib as mpl
#
# ## agg backend is used to create plot as a .png file
# mpl.use('agg')

import matplotlib.pyplot as plt

# ## Create data
# np.random.seed(10)
# collectn_1 = np.random.normal(100, 10, 200)
# collectn_2 = np.random.normal(80, 30, 200)
# collectn_3 = np.random.normal(90, 20, 200)
# collectn_4 = np.random.normal(70, 25, 200)
#
# ## combine these different collections into a list
# data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]
#
# # Create a figure instance
# fig = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax = fig.add_subplot(111)
#
# # Create the boxplot
# bp = ax.boxplot(data_to_plot)
#
# # Save the figure
# fig.savefig('fig1.png', bbox_inches='tight')

mean_teeth = [
    2.33,
    1.84,
    1.86,
    1.56,
    1.39,
    1.23,
    1.59,
    1.55,
    1.84,
    1.43,
    1.98,
    2.83,
    2.05,
    1.95,
    2.92,
    2.36,
    1.75,
    2.34,
    2.04,
    2.2,
    1.85,
    1.69,
    1.86,
    1.63,
    1.7,
    1.48,
    2.12,
    3.35,
    1.84,
    1.8,
    1.97,
    2.62,
    2.33,
    2.07,
    2.58,
    1.8,
    1.52,
    1.61,
    1.68,
    1.75,
    2.25,
    1.88,
    2,
    1.72,
    1.78,
    2.01,
    1.91,
    3.28,
    2.52,
    2.82,
    3.22,
    3.07,
    2.11,
    2.65,
    1.75,
    1.92,
    2.08,
    3.7,
    1.94,
    1.81,
    2.82,
    1.93,
    2.33,
    3.06,
    2.05,
    1.98,
    1.92,
    2.21
    ]

median_teeth = [
    1.67,
    1.66,
    1.07,
    1.02,
    1.01,
    0.87,
    1.15,
    1.22,
    1.33,
    1,
    1.24,
    1.16,
    1.11,
    1.23,
    1.38,
    1.19,
    1.21,
    1.28,
    1.8,
    1.83,
    1.27,
    0.93,
    1.33,
    1.16,
    1.21,
    1.26,
    1.02,
    1.07,
    1.31,
    1.15,
    1.3,
    1.21,
    1.58,
    1.6,
    2.1,
    1.35,
    0.89,
    0.8,
    1.02,
    0.76,
    0.98,
    0.94,
    1.4,
    1.05,
    0.9,
    1.01,
    1.14,
    1.43,
    1.26,
    1.31,
    1.76,
    1.76,
    1.7,
    1.93,
    0.99,
    1.07,
    1.14,
    1.13,
    1.07,
    1.26,
    1.36,
    1.24,
    1.32,
    1.03,
    1.31,
    1.2,
    1.28,
    1.18,
    ]

def plot_68_teeth_landmarks():

    colors = np.arange(len(mean_teeth))
    x = [0, 4]
    y = [0, 4]
    plt.scatter(mean_teeth, median_teeth, c=colors)
    plt.title('Mean error VS. Median error')
    plt.plot(x,y)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.xlabel('Mean error (mm)')
    plt.ylabel('Median error (mm)')
    plt.show()


mean_bony = [
        2.64,
        2.77,
        2.25,
        1.93,
        1.67,
        5.21,
        4.44,
        3.55,
        3.75,
        1.76,
        1.92,
        2.6,
        3.23,
        3.19,
        2.82,
        3.17,
        3.16,
        2.36,
        1.98,
        3.19,
        2.02,
        3,
        3.43,
        1.61,
        3.25,
        2.99,
        3.68,
        3.76,
        2.94,
        4.9,
        1.88,
        1.67,
        2.39,
        2.56,
        2.33,
        2.58,
        2.76,
        2.17,
        2.97,
        2.54,
        2.83,
        3.28,
        3.05,
        3.63,
        3.23,
        2.82,
        3.28,
        3.24,
        2.47,
        3.32,
        2.88,
        2.47,
        6.04,
        6.25,
        3.08,
        3.06,
        2.73,
        2.27,
        2.96,
        3.15,
        2.89,
        3.84,
        3.52,
        2.58,
        4.62,
        3.68,
        2.61,
        2.86,
        3.11
    ]

median_bony = [
        2.67,
        2.47,
        1.93,
        1.72,
        1.55,
        5.29,
        4.47,
        2.86,
        3.17,
        1.75,
        1.85,
        2.35,
        2.75,
        2.93,
        2.57,
        2.45,
        2.38,
        2.08,
        1.77,
        2.91,
        2.05,
        2.64,
        3.11,
        1.42,
        2.72,
        2.64,
        3.44,
        4.29,
        2.45,
        2.27,
        1.75,
        1.6,
        2.28,
        2.38,
        2.24,
        2.33,
        2.31,
        2.06,
        2.11,
        1.91,
        2.66,
        2.64,
        2.55,
        3.7,
        2.95,
        2.64,
        2.73,
        2.75,
        2.31,
        3.18,
        2.58,
        2.4,
        3.69,
        5.23,
        2.79,
        2.66,
        2.55,
        1.97,
        2.32,
        3,
        2.52,
        3.32,
        2.81,
        2.05,
        3.55,
        2.58,
        2.51,
        2.64,
        2.73
    ]

def plot_66_bony_landmarks():

    colors = np.arange(len(mean_bony))
    x = [0, 4]
    y = [0, 4]
    plt.scatter(mean_bony, median_bony, c=colors)
    plt.title('Mean error VS. Median error')
    plt.plot(x,y)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.xlabel('Mean error (mm)')
    plt.ylabel('Median error (mm)')
    plt.show()


mean_facial = [
        2.04,
        2.73,
        2.36,
        2.38,
        1.98,
        2.14,
        2.22,
        2.45,
        2.45,
        2.02,
        2.18,
        1.89,
        1.93,
        3.19,
        2.59,
        2.31,
        2.51,
        6.7,
        2.31,
        2.36,
        4.32,
        3.09,
        2.94,
        2.11,
        1.43,
        1.61,
        2.08,
        2,
        2.44,
        3.09,
        3.85,
        1.53,
        2.97,
        1.75,
        1.73,
        2.58,
        2.9,
    ]

median_facial = [
        1.72,
        1.79,
        2.32,
        2.45,
        2.06,
        2.22,
        2.1,
        1.94,
        1.82,
        1.85,
        2.16,
        1.94,
        1.86,
        2.86,
        2.18,
        1.78,
        2.21,
        4.69,
        2.25,
        2.35,
        4.12,
        2.39,
        2.56,
        1.93,
        1.57,
        1.44,
        1.84,
        2.15,
        2.01,
        2.55,
        3.22,
        1.34,
        3.13,
        1.8,
        1.48,
        2.51,
        2.07,
    ]

def plot_41_facial_landmarks():

    colors = np.arange(len(mean_facial))
    x = [0, 4]
    y = [0, 4]
    plt.scatter(mean_facial, median_facial, c=colors)
    plt.title('Mean error VS. Median error')
    plt.plot(x,y)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.xlabel('Mean error (mm)')
    plt.ylabel('Median error (mm)')
    plt.show()


def plot_merge():
    colors = np.arange(len(mean_facial))
    x = [0, 4]
    y = [0, 4]
    plt.scatter(mean_facial, median_facial, color='green')
    plt.scatter(mean_bony, median_bony, color='yellow')
    plt.scatter(mean_teeth, median_teeth, color='blue')

    plt.title('Mean error VS. Median error')
    plt.plot(x,y)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.xlabel('Mean error (mm)')
    plt.ylabel('Median error (mm)')
    plt.show()


# plot_66_bony_landmarks()
# plot_68_teeth_landmarks()
# plot_41_facial_landmarks()
plot_merge()
