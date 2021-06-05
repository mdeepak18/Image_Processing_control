import numpy as np
import cv2

img = cv2.imread("Resources/plum.bmp")
print(img.shape)
img2 = img.reshape((-1, 3))
img2 = np.float32(img2)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# clusters
k = 2
attempts = 10
ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
print(center)
center = np.uint8(center)
# print(img2.shape)
print(center.shape)
print(label.flatten())
print(label.flatten().shape)
res = center[label.flatten()]
print(res)
res2 = res.reshape((img.shape))
cv2.imwrite("clustered.jpg", res2)
