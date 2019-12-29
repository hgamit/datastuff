from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import tqdm
import pandas as pd
import numpy as np
import cv2

TRAIN_PATH = '/content/openimages2019-train-f/train_samples/train_samples/'
MASK_PATH = '/content/train-masks-f/'


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
                                     

def rle2mask(height, width, encoded):
    img = np.zeros(height*width, dtype=np.uint8)

    if isinstance(encoded, float):
        return img.reshape((width, height)).T

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


def resize_label(filtered_df):
    EncodedPixels = np.array([])
    for index, row in filtered_df.iterrows():
        #pdb.set_trace()
        mask_decoded = cv2.imread(os.path.join(MASK_PATH, row['MaskPath']))
        mask = cv2.resize(mask_decoded, (256, 256), interpolation=cv2.INTER_NEAREST)
        s2 = mask2rle(mask)
        EncodedPixels = np.append(EncodedPixels, s2)
    
    return EncodedPixels


INPUT_PATH = f'data/train.ver0.csv'
OUTPUT_PATH = f'data/train_256.ver0.csv'

df = pd.read_csv(INPUT_PATH)
df['EncodedPixels'] = resize_label(df)
df.to_csv(OUTPUT_PATH, index=False)