import os
from PIL import Image, ImageFile
import collections
import numpy as np 
import cv2
import torch
import matplotlib.pyplot as plt 
from pathlib import Path
import seaborn as sns
IMAGE_SIZE = (512, 512)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
    shape: (height,width) of array to return
    Returns numpy array according to the shape, 1 - mask, 0 - background
    '''
    shape = (shape[1], shape[0])
    s = mask_rle.split()
    # gets starts & lengths 1d arrays
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # sets mark pixles
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image
    return img.reshape(shape).T

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)
        
def get_unique_class_id_df(inital_df):
    temp_df = inital_df.groupby(['ImageId','ClassId'])['EncodedPixels'].agg(lambda x: ' '.join(list(x))).reset_index()
    size_df = inital_df.groupby(['ImageId','ClassId'])[['Height', 'Width']].mean().reset_index()
    temp_df = temp_df.merge(size_df, on=['ImageId','ClassId'], how='left')
    
    return temp_df