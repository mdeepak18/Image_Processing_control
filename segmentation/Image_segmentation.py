import numpy as np
import time
import cv2

image = cv2.imread("Resources/plum.bmp")
mask = np.zeros(image.shape[:2], dtype="uint8")

#manually determined box using Paint application
rect = (10, 82, 210, 115)

fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")
iter=10

start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel, fgModel, iter, mode=cv2.GC_INIT_WITH_RECT)
end = time.time()
print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
outputMask = (outputMask * 255).astype("uint8")
output = cv2.bitwise_and(image, image, mask=outputMask)

cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imwrite("plum_mask.png", outputMask)
cv2.imshow("GrabCut Output", output)
#cv2.imwrite("segmented_plum.jpg", output)
cv2.waitKey(1000)

#using the mask output from above code, I modified it on photoshop, a bit, to tell the machine that some part of
#foreground and background were mislabelled. This can be done interactively during runtime as well or using Neural Network algorithms as well.
#so the next line of code is the modified mask made using interactive Grabcut implementation.

mask = cv2.imread("plum_mask_p.png", cv2.IMREAD_GRAYSCALE)

roughOutput = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("output with modified mask", roughOutput)
cv2.waitKey(0)

mask[mask > 0] = cv2.GC_PR_FGD
mask[mask == 0] = cv2.GC_BGD
fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")
iter=10
start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel, fgModel, iter, mode=cv2.GC_INIT_WITH_MASK)
end = time.time()
print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

#uncomment the following 2 blocks to see the foreground, probale foreground, background and probable background

#values = (
#	("Definite Background", cv2.GC_BGD),
#	("Probable Background", cv2.GC_PR_BGD),
#	("Definite Foreground", cv2.GC_FGD),
#	("Probable Foreground", cv2.GC_PR_FGD),
#)

#for (name, value) in values:
#	print("[INFO] showing mask for '{}'".format(name))
#	valueMask = (mask == value).astype("uint8") * 255
#	cv2.imshow(name, valueMask)
#	cv2.waitKey(0)

outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
outputMask = (outputMask * 255).astype("uint8")

output = cv2.bitwise_and(image, image, mask=outputMask)

cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.imwrite("plum_output.jpg",output)
cv2.waitKey(0)