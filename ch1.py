# import cv2

# print("package imported")

# img=cv2.imread("Resources/plum.bmp")
# cv2.imshow("Disp",img)
# cv2.waitKey(2000)

# vid = cv2.VideoCapture("Resources/kitty.mp4")
# while True:
#    success, img = vid.read()
#   cv2.imshow("Disp", img)
#  cv2.waitKey(500)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#  break

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 100)

# while True:
#    success, img = cap.read()
#   cv2.imshow("Disp", img)
#   # cv2.waitKey(500)
#  if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

import cv2

img = cv2.imread("Resources/plum.bmp")
#imgGr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#imgBlur = cv2.GaussianBlur(img,(5,5),0)
imgCanny = cv2.Canny(img,100,100)
#print(img.shape)
#imgcp = cv2.resize(img,(100,200))
#imgcp = img[0:100,100:200]
cv2.imshow("Dispgray",imgCanny)
cv2.imwrite("plum_edges.jpg", imgCanny)
#cv2.imshow("Display",imgBlur)
#cv2.imshow("DIsplay",imgCanny)
cv2.waitKey(0)
