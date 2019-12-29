import cv2 as cv2
import numpy as np


sz = 256
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def equalize_hist(input_path):
    img = cv.imread(input_path)
    for c in range(0, 2):
        img[:,:,c] = cv.equalizeHist(img[:,:,c])
    blur = cv.blur(img,(5,5))
    diff=img-blur
    cv.imshow('Histogram equalized', diff)
    cv.waitKey(0)
    cv.destroyAllWindows()

def clahe_rgb(input_path):
    bgr = cv.imread(input_path)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    gridsize = 5
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv.merge(lab_planes)
    bgr2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    blur = cv.blur(bgr2,(5,5))
    diff=bgr2-blur
    cv.imshow('CLAHE RGB', diff)
    cv.waitKey(0)
    cv.destroyAllWindows()


def clahe_greyscale(input_path):
    img = cv.imread(input_path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_image)    
    blur = cv.blur(cl1,(5,5))
    diff=cl1-blur
    cv.imshow('CLAHE Grayscale', diff)
    cv.waitKey(0)
    cv.destroyAllWindows()
import math

def center_crop(im, min_sz=None):

    """ Returns a center crop of an image"""
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    start_r = math.ceil((r-min_sz)/2)
    start_c = math.ceil((c-min_sz)/2)
    return crop(im, start_r, start_c, min_sz)

def crop(im, r, c, sz): return im[r:r+sz, c:c+sz]

filename = f"C:/Users/hmnsh/Downloads/resized-2015-2019-blindness-detection-images/good/6a2642131e4a.jpg"

image = cv2.imread(filename)
print(image.shape) #Number of rows, column, channel
print(image.size) # total number of pixels
print(image.dtype) # datatype
#b, g, r = cv2.split(img)
#img = cv2.merge((b, g, r))
#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #COLOR_RGB2BGR, COLOR_RGB2GRAY
#image = cv2.convertScaleAbs(image, alpha=1.4, beta=0.5)
# draw in the image
image = cv2.resize(image, (320, 320))
imageC = center_crop(image)
#image = crop_image_from_gray(image)

#k = np.max(image.shape)/10
#imageG=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , k) ,-4 ,50)
#print(image)
cv2.imshow("Updated",np.hstack((image, imageC))) 
#n_white_pix = np.sum(image == 255)
#print('Number of white pixels:', n_white_pix)
#cv2.imshow('image', image)
cv2.waitKey(5000)
#cv2.destroyAllWindows()

