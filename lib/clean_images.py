#!/usr/bin/env python
# coding: utf-8

from PIL import Image, ImageOps
from scipy import misc

import glob
import numpy as np
import os
import pickle
import random
import threading
import torch
import torchvision.transforms as transforms


def resize_image(image, min_size, da=False):
    """Resize if needed.

    Args:
        image (array of int): The image to be resized.
        min_size (int): The size of the smallest side of the image.

    Returns:
        image (array of int): The resized image.
    """

    height = image.height
    width = image.width

    if da:
        height = int(height*(1+0.15*np.random.randn()))
        width = int(width*(1+0.15*np.random.randn()))
        image = image.resize(size=(width, height))
    
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


def trans(image):
    if np.random.rand() > 0.5:
        image = image[:, ::-1, :]

    return image

def color(image):
    if np.random.rand() > 0.5:
        rgb = [[1.+np.random.randn(3)*0.15]]
        image *= rgb
    return image


def square_crop_resize(bytes, img_size, da=False):
    # Read the image.
    image = Image.open(bytes)
    image = ImageOps.autocontrast(image)

    # Resize if needed.
    image = np.array(resize_image(image, img_size, da))/255.

    # Convert the image to RGB if needed.
    image = convert2rgb(image)

    if da:
        image = trans(image)
        image = color(image)
    
    # Get a square-crop of the image.
    height = len(image)
    width = len(image[0])
    
    smallest_is_width = True if height>width else False

    if smallest_is_width:
        h_remaining_space = height - img_size
        if da:
            offset = np.random.randint(h_remaining_space+1)
        else:
            offset = 0
        image = image[offset:img_size+offset,:,:3]
    else:
        w_remaining_space = width - img_size
        if da:
            offset = np.random.randint(w_remaining_space+1)
        else:
            offset = w_remaining_space//2
        image = image[:,offset:img_size+offset,:3]
    
    return image