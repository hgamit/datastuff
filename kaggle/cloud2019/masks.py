import numpy as np
import cv2
import matplotlib.pyplot as plt

masks = np.zeros((350, 525, 1), dtype=np.float32)
mask = cv2.imread("Fish0011165.jpg")
img = cv2.imread("0011165.jpg", 0)
#img = cv2.bitwise_not(img)
ret,thresh1 = cv2.threshold(img,115,255,cv2.THRESH_BINARY)
thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB) #cv2.COLOR_BGR2GRAY
if mask[:, :, 0].shape != (350, 525):
    mask = cv2.resize(mask, (525, 350))

masks[:, :, 0] = mask[:, :, 0]
masks = masks/255


height, width = img.shape[:2]
masks = cv2.resize(masks, (width, height), interpolation=cv2.INTER_NEAREST)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,10))
ax[0, 0].imshow(img, cmap = plt.cm.gray)
  #ax[1].imshow(mask_decoded, alpha=0.3)
  # visualize the image and map
ax[0, 1].imshow(img, cmap = plt.cm.gray)
ax[0, 1].imshow(masks, alpha=0.6, cmap='Reds')
ax[1, 0].imshow(thresh1, cmap = plt.cm.gray)
ax[1, 1].imshow(thresh1, cmap = plt.cm.gray)
ax[1, 1].imshow(masks, alpha=0.6, cmap='Reds')