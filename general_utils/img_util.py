import os
import cv2
import numpy as np
import math
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

def read_image(img_file):
    '''
    This function reads the image file using cv2
    img_file: path to the image file
    return : the image
    '''
    img = cv2.imread(img_file)
    return img


def crop_obj_double_bb(img, bbox):
    '''
    This function crops the image around the given bounding box with double scale
    img : the input image
    bbox : the given bounding box in [x, y, w, h] format
    return : the new cropped image
    '''
    (img_height, img_width, _) = np.shape(img)
    new_l = int(max(bbox[2], bbox[3]))
    [d_width, d_height] = [new_l*2, new_l*2]
    center_x, center_y = bbox[0] + bbox[2]/2 , bbox[1] + bbox[3]/2
    bb_cx, bb_cy, bb_w, bb_h = int(center_x-(d_width/2)), int(center_y-(d_height/2)), int(d_width), int(d_height)
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox)
    else:
        img = img[bb_cy:bb_cy+d_height, bb_cx:bb_cx+d_width, :]
        return img





def rotate(image, angle):
    """
    This function rotates an image about it's centre by the given angle (in degrees).
    The returned image will be large enough to hold the entire new image, with a black background.
    image: the input image
    angle: the angle of rotation
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    this function computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle, given a rectangle of size wxh that has been rotated by 'angle' (in
    radians).
    w: the width of the rectangle
    h: the height of the rectangle
    angle: the angle of rotation
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    This function crops the image to the given width and height around it's centre point.
    image: the input image
    height: the desired height
    width: the desired width
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def crop_largest_rectangle(image, angle, height, width):
    """
    This function crops around the center the largest possible rectangle.
    found with largest_rotated_rect.
    image: the input image
    angle: the angle of rotation
    height: the height of the rectangle
    width: the width of the rectangle
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """
    return crop_around_center(
        image,
        *largest_rotated_rect(
            width,
            height,
            math.radians(angle)
        )
    )

def generate_rotated_image(image, angle, size=None, crop_center=False,
                           crop_largest_rect=False):
    """
    This function generates a valid rotated image for the DataGenerator.
    image: the input image
    angle: the angle of rotation
    size: the desired size of the final image to be resized
    crop_center: If the image is rectangular, the crop_center option should be used to make it square.
    crop_largest_rect: To crop out the black borders after rotation, use the crop_largest_rect option.
    return : the rotated image
    Source: https://github.com/d4nst/RotNet/tree/master
    license: MIT License
    """
    height, width = image.shape[:2]
    if crop_center:
        if width < height:
            height = width
        else:
            width = height

    image = rotate(image, angle)

    if crop_largest_rect:
        image = crop_largest_rectangle(image, angle, height, width)

    if size:
        image = cv2.resize(image, size)

    return image


def resize_with_border(img, desired_size, resize=True):
    """
    This function resizes the image to the desired size with keeping the aspect-ratio by adding a border around it.
    img: the input image
    desired_size : the desired size of the final image
    resize: If True, It resizes the image to the desired size
    return : the resized image with black borders
    """
    old_size = img.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size / max(old_size))
    new_size = tuple([int(x * ratio) for x in old_size])
    if resize:
        img = cv2.resize(img, (new_size[1], new_size[0]))
    #else:
    #    ratio = 1.0
    #    new_size = tuple([int(x * ratio) for x in old_size])

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


def crop_image_from_center(img, des_size):
    '''
    This function crops the image from the center point of the image
    img : the input image
    des_size : the desired size of the cropped area
    return : the new cropped image from center
    '''
    (d_width, d_height) = des_size
    image_size = (img.shape[0], img.shape[1])
    image_center = tuple(np.array(image_size) / 2)
    img = img[int(image_center[0] - (d_width / 2)):int(image_center[0] + (d_width / 2)),
    int(image_center[1] - (d_height / 2)):int(image_center[1] + (d_height / 2))]
    return img





def display_image_grid(images, n=10, angles=None, output_path="None"):
    '''
    This function visualizes the given images in a grid with labels
    Source: https://www.kaggle.com/code/alibalapour/rotation-prediction-de-skewing-text-in-images/notebook
    license: Apache 2.0
    '''
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n, n),
                     axes_pad=0.25,
                     )

    i = 0
    for ax, im in zip(grid, images):
        ax.imshow(im, cmap='gray');
        ax.set_xticks([])
        ax.set_yticks([])
        if angles is not None:
            angle = angles[i]
            ax.set_title(label=str(angle))
        i += 1

    output_img = "/".join(list(output_path.split("/")[0:-1])) + "/train_batch.png"
    plt.savefig(output_img)
    plt.close()
    #plt.show()
