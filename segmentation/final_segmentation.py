import numpy as np
import argparse
import time
import cv2
import os



image = cv2.imread("Resources/plum.bmp")
mask = cv2.imread("plum_mask_p.png", cv2.IMREAD_GRAYSCALE)

roughOutput = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Rough Output", roughOutput)
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

values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)

for (name, value) in values:
	print("[INFO] showing mask for '{}'".format(name))
	valueMask = (mask == value).astype("uint8") * 255
	cv2.imshow(name, valueMask)
	cv2.waitKey(0)

outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
outputMask = (outputMask * 255).astype("uint8")

output = cv2.bitwise_and(image, image, mask=outputMask)

cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.imwrite("plum_output.jpg",output)
cv2.waitKey(0)