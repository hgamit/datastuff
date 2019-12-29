import numpy as np
import cv2
from matplotlib import pyplot as plt


def equalize_hist(input_path):
    img = cv2.imread(input_path)
    for c in range(0, 2):
        img[:,:,c] = cv2.equalizeHist(img[:,:,c])

    cv2.imshow('Histogram equalized', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clahe_rgb(input_path):
    bgr = cv2.imread(input_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    gridsize = 5
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imshow('CLAHE RGB', bgr2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clahe_greyscale(input_path):
    img = cv2.imread(input_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_image)    
    cv2.imshow('CLAHE Grayscale', cl1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted (im, 4, bg, -4, 128)

filename = f"C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/good/c56293f53191.jpg"

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = subtract_median_bg_image(img)
dst1 = cv2.fastNlMeansDenoisingColored(dst,None,6,6,7,21)
#dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
#k = np.max(img.shape)/10
#gaus_img = cv2.GaussianBlur( img , (59,59) , k)
# Blur the image
#blurred = cv2.blur(img, ksize=(15, 15))
#dst=cv2.addWeighted(img,4, bg ,-4 ,50)
#dst1 = cv2.bilateralFilter(dst, 9, 75, 75, cv2.BORDER_DEFAULT)
#th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
dst1 = cv2.fastNlMeansDenoisingColored(dst,None,6,6,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst1)
plt.show()

#equalize_hist(filename)