import cv2
import matplotlib.pyplot as plt


def show_two_images(img, dst, left_title='', right_title=''):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    plot_on_axis(ax1, img, left_title)
    plot_on_axis(ax2, dst, right_title)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def show_four_images(imgs):
    f, axes = plt.subplots(2, 2, figsize=(24, 9))
    plot_on_axis(axes[0, 0], imgs[0], 'undistorted image')
    plot_on_axis(axes[0, 1], imgs[1], 'mask')
    plot_on_axis(axes[1, 0], imgs[2], 'warped mask')
    plot_on_axis(axes[1, 1], imgs[3], 'lane line fitting')
    plt.show()


def plot_on_axis(axis, img, title):
    if len(img.shape) == 3 and img.shape[2] == 3:
        axis.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        axis.imshow(img, cmap='gray')
    axis.set_title(title)


def perspective_transform_result(img, dst, src_points, dst_points, left_title='', right_title=''):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    plot_on_axis(ax1, img, left_title)
    ax1.plot(src_points[:, 0], src_points[:, 1], 'r-', lw=5)
    ax1.set_title(left_title, fontsize=50)
    plot_on_axis(ax2, dst, right_title)
    ax2.plot(dst_points[:, 0], dst_points[:, 1], 'r-', lw=5)
    ax2.set_title(right_title, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
