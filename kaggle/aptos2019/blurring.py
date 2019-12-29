import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = f"C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/good/c56293f53191.jpg"
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5, 5))
k = np.max(img.shape)/10
gblur = cv2.GaussianBlur(img, (5,5) , k)
#gblur = cv2.addWeighted ( img,2, cv2.GaussianBlur( img , (0,0) , k) ,-2 ,70)
median = cv2.medianBlur(img, 5)
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)


titles = ['image', 'GaussianBlur']
images = [img, gblur]

for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()