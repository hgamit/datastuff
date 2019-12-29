#!/usr/bin/python

# pip install lxml

import sys
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np 
import cv2
import json
import itertools
from imantics import Polygons, Mask
import matplotlib.pyplot as plt
from matplotlib import patches as patches

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def rle_to_mask(rle_string, height, width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 0
        img = img.reshape(cols,rows)
        img = img.T
        return img

def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    #print("rows", rows)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def rle2mask(mask_rle, shape=(2100,1400), shrink=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T[::shrink,::shrink]

def rle2bb(rle):
    if rle=='': return (0,0,0,0)
    mask = rle2mask(rle)
    z = np.argwhere(mask==1)
    mn_x = np.min( z[:,0] )
    mx_x = np.max( z[:,0] )
    mn_y = np.min( z[:,1] )
    mx_y = np.max( z[:,1] )
    return (mn_x,mn_y,mx_x-mn_x,mx_y-mn_y)

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None

def convert(df, json_file, img_dir):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = {name: i for i, name in enumerate(df['Label'].unique())}
    bnd_id = START_BOUNDING_BOX_ID
    for index, row in df.iterrows():
        if row['EncodedPixels'] != -1:
        ## The filename must be a number      
            image_id = row['ImageId']
            filename = os.path.join(img_dir, row['ImageId'])
            height, width s= cv2.imread(filename).shape[:2]
            image = {
                "file_name": image_id,
                "height": height,
                "width": width,
                "id": image_id,
            }
            json_dict["images"].append(image)
            ## Currently we do not support segmentation.
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            category = row['Label']
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            
            mask_decoded = rle_to_mask(row['EncodedPixels'], height, width)
            rows = np.any(mask_decoded, axis=1)
            cols = np.any(mask_decoded, axis=0)
            #print("rows", rows)
            xmin, xmax = np.where(rows)[0][[0, -1]]
            ymin, ymax = np.where(cols)[0][[0, -1]]
            
            polygons = Mask(mask_decoded).polygons()
            #poly = [(x + 0.5, y + 0.5) for x, y in zip(rows, cols)]
            #poly = list(itertools.chain.from_iterable(poly))
            #xmin, xmax, ymin, ymax = bounding_box(mask_decoded)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": int(o_width * o_height),
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [int(xmin), int(ymin), int(o_width), int(o_height)],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [polygons.segmentation],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def plot_cloud(img_path, img_id, label_mask):
    img = cv2.imread(os.path.join(img_path, img_id))
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    ax[0].imshow(img)
    ax[1].imshow(img)
    cmaps = {'Fish': 'Blues', 'Flower': 'Reds', 'Gravel': 'Greys', 'Sugar':'Purples'}
    colors = {'Fish': 'Blue', 'Flower': 'Red', 'Gravel': 'Gray', 'Sugar':'Purple'}
    for label, mask in label_mask:
        mask_decoded = rle_to_mask(mask, img.shape[0], img.shape[1])
        if mask != -1:
            rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
            bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor=colors[label],facecolor='none')
            ax[0].add_patch(bbox)
            ax[0].text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))
            ax[1].imshow(mask_decoded, alpha=0.3, cmap=cmaps[label])
            ax[0].text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))


if __name__ == "__main__":
    
    data_dir = 'C:/Users/hmnsh/repos/datastuff/kaggle/cloud2019/'
    train_image_path = os.path.join(data_dir, 'clouds_resized/train_images_525/')
    #test_image_dir = os.path.join(ship_dir, 'test_v2')
    #Get target label and image name in separate column
    train_df = pd.read_csv(f"{data_dir}/train.csv", nrows=10).fillna(-1)
    # image id and class id are two seperate entities and it makes it easier to split them up in two columns
    train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
    train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
    # lets create a dict with class id and encoded pixels and group all the defaults per image
    train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)
    train_df = train_df[train_df['EncodedPixels'] != -1]
    
    img = cv2.imread(os.path.join(train_image_path, train_df['ImageId'][1]))
    mask_decoded = rle2mask(train_df['Label_EncodedPixels'][0][1])#, img.shape[0], img.shape[1])
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    ax[0].imshow(img)
    ax[1].imshow(mask_decoded)
    
    # lets group each of the types and their mask in a list so we can do more aggregated counts
    grouped_EncodedPixels = train_df.groupby('ImageId')['Label_EncodedPixels'].apply(list)
    
    for image_id, label_mask in grouped_EncodedPixels.sample(2).iteritems():
        plot_cloud(train_image_path, image_id, label_mask)
        
    
    json_file = data_dir + "json_file.json"
    print("Converting: {}".format(json_file))
    convert(train_df, json_file, train_image_path)
    print("Success: {}".format(json_file))