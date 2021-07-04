import math
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np


def irregular_mask(image_height, image_width, batch_size=1, min_strokes=16, max_strokes=48):
    masks = []

    for b in range(batch_size):
        mask = np.zeros((image_height, image_width), np.uint8)
        mask_shape = mask.shape

        max_width = 20
        number = random.randint(min_strokes, max_strokes)
        for _ in range(number):
            model = random.random()
            if model < 0.6:
                # Draw random lines
                x1, x2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[0])
                y1, y2 = random.randint(1, mask_shape[1]), random.randint(1, mask_shape[1])
                thickness = random.randint(4, max_width)
                cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

            elif model > 0.6 and model < 0.8:
                # Draw random circles
                x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                radius = random.randint(4, max_width)
                cv2.circle(mask, (x1, y1), radius, (1, 1, 1), -1)

            elif model > 0.8:
                # Draw random ellipses
                x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                s1, s2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
                thickness = random.randint(4, max_width)
                cv2.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

        masks.append(mask[:, :, np.newaxis])

    return np.array(masks).astype("float32")


def center_mask(image_height, image_width, batch_size=1):
    mask = np.zeros((batch_size, image_height, image_width, 1)).astype("float32")
    mask[:, image_height // 4 : (image_height // 4) * 3, image_height // 4 : (image_height // 4) * 3, :] = 1.0

    return mask


def generate_rectangle_size(randomize_size=True, min_size=64, max_size=128, square=False):
    """
    Return dimensions of a rectangle
    :param randomize_size: If rectangle should have a random size, when False, the max_size will be used
    :param min_size: min size of the rectangle
    :param max_size: max size of the rectangle
    :param square: if width and height should be equal
    :return:
    """
    width = random.randint(min_size, max_size) if randomize_size else max_size
    height = width if square or not randomize_size else random.randint(min_size, max_size)
    return width, height


def place_rectangle(image_height, image_width, width, height, position="uniform"):
    """
    Generates an x, y coordinate where the mask with given width and height can be places
    :param image_height:
    :param image_width:
    :param width:
    :param height:
    :param position: Where the mask should be placed, 'uniform' = random uniform towards center, center = center, random = randomly on image
    :return: x, y coordinates
    """
    if position == "uniform":
        pos_x = math.floor(np.random.uniform(image_width - width))
        pos_y = math.floor(np.random.uniform(image_height - height))
    elif position == "center":
        pos_x = image_width // 2 - width // 2
        pos_y = image_height // 2 - height // 2
    elif position == "random":
        pos_x = np.random.randint(0, high=image_width - width)
        pos_y = np.random.randint(0, high=image_height - height)
    else:
        raise Exception(f"Unsupported rectangle position '{position}'")

    return pos_x, pos_y


def rectangle_mask(image_height, image_width, batch_size=1, shape="random_rect", position="uniform"):
    """
    Generate a rectangular style mask
    :param image_height:
    :param image_width:
    :param batch_size:
    :param shape: Shape of the rectangle (square)?_(random)?_\\d+_\\d+
    :param position: Where the mask should be placed, 'uniform' = random uniform towards center, center = center, random = randomly on image
    :return:
    """
    square = "square" in shape
    randomize_size = "random" in shape
    # The idea is that a user can specify something like random_rect_64_32, square_64
    sizes = re.findall(r"\d+", shape)
    max_size = int(sizes[0]) if len(sizes) > 0 else 128
    min_size = int(sizes[1]) if len(sizes) > 1 else 64

    rect_size = generate_rectangle_size(randomize_size=randomize_size, square=square, max_size=max_size, min_size=min_size)
    rect_pos = place_rectangle(image_height, image_width, rect_size[0], rect_size[1], position=position)

    mask = np.zeros((batch_size, image_height, image_width, 1)).astype("float32")
    mask[:, rect_pos[0] : rect_pos[0] + rect_size[0], rect_pos[1] : rect_pos[1] + rect_size[1], :] = 1.0

    return mask


def json_data_mask(image_height, image_width, batch_size=1, image_name=None, mask_data=None):
    # FIXME: Figure this out, maybe create maks at load time?
    mask = np.zeros((batch_size, image_height, image_width, 1)).astype("float32")
    mask[:, mask_data[0] : mask_data[2], mask_data[1] : mask_data[3], :] = 1.0

    return mask


def save_images(input_image, ground_truth, prediction_coarse, prediction_refine, path):

    display_list = [input_image, ground_truth, prediction_coarse, prediction_refine]
    img = np.concatenate(display_list, axis=1)
    plt.imsave(path, np.clip(img, 0, 1.0))
