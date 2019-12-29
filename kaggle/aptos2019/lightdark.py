import cv2
import numpy as np

filename = f"C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/blurry/3006_right.jpg"
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
print(np.max(cv2.convertScaleAbs(cv2.Laplacian(img,3))))