#!/usr/bin/env python
# coding: utf-8

from PIL import Image
from scipy import misc

import glob
import numpy as np
import os
import pickle
import random
import threading


def resize_image(image, min_size):
    """Resize if needed.

    Args:
        image (array of int): The image to be resized.
        min_size (int): The size of the smallest side of the image.

    Returns:
        image (array of int): The resized image.
    """

    height = image.height
    width = image.width
    
    if ((height != min_size and width != min_size) or
        (height == min_size and width < min_size) or
        (width == min_size and height < min_size)):

        if height < width:
            output_size = (int(width * min_size / height), min_size)
        elif height == width:
            output_size = (min_size, min_size)
        else:
            output_size = (min_size, int(height * min_size / width))
        image = np.asarray(image.resize(size=output_size))

    return image


def convert2rgb(image):
    """Convert to RGB if needed.

    Args:
        image (array of int): The image to be colored.

    Returns:
        image (array of int): The colored image.
    """

    height = len(image)
    width = len(image[0])
    
    try:
        len(image[0][0]) == 3
    except TypeError:
        tmp_img = np.zeros((height, width, 3), dtype=np.uint8)
        if len(np.shape(image)) == 2:
            tmp_img[:, :, 0] = image
        elif len(np.shape(image)) > 2:
            tmp_img[:, :, 0] = image[:,:,0]
        tmp_img[:, :, 1] = tmp_img[:, :, 2] = tmp_img[:, :, 0]
        image = tmp_img
    return image


def square_crop_resize(path, img_size):
    # Read the image.
    image = Image.open(path)

    # Resize if needed.
    image = resize_image(image, img_size)

    # Convert the image to RGB if needed.
    image = convert2rgb(image)
    
    # Get a square-crop of the image.
    height = len(image)
    width = len(image[0])
    
    smallest_is_width = True if height>width else False

    if smallest_is_width:
        h_remaining_space = height - img_size
        offset = np.random.randint(h_remaining_space+1)
        image = image[offset:img_size+offset,:,:3]
    else:
        w_remaining_space = width - img_size
        offset = np.random.randint(w_remaining_space+1)
        image = image[:,offset:img_size+offset,:3]
    
    return image