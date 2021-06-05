import numpy as np
import cv2

img = cv2.imread("Resources/plum.bmp")

imgCanny = cv2.Canny(img,100,100)
cv2.imshow("Disp_canny",imgCanny)
cv2.imwrite("plum_more_edges.jpg", imgCanny)

#clustering image by K-means Clustering

img2 = img.reshape((-1, 3))
img2 = np.float32(img2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# clusters
k = 2
attempts = 10
ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
#print(center)
center = np.uint8(center)
# print(img2.shape)
#print(center.shape)
#print(label.flatten())
#print(label.flatten().shape)
res = center[label.flatten()]
#print(res)
res2 = res.reshape((img.shape))
cv2.imshow("plum_clustered",res2)

imgCanny = cv2.Canny(res2,100,100)
cv2.imshow("Dispgray",imgCanny)
cv2.imwrite("plum_less_edges.jpg", imgCanny)
cv2.waitKey(0)