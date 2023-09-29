# Загрузка установленных библиотек
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import argparse
import scipy
import imageio
import skimage.segmentation
import scipy.signal as sig
import imutils
from PIL import Image, ImageEnhance, ImageFilter


# Загрузка исходного изображения для обработки и поиска дефектов
img = Image.open("/Users/ivansvininnikov/PycharmProjects/Course_projects/images/006450.png")

# Предобработка изображения
img = img.filter(ImageFilter.GaussianBlur(3))

enhancer_2 = ImageEnhance.Brightness(img)
factor = 12
im_output = enhancer_2.enhance(factor)

im_output.save('constrast_image_project.png')

img = cv2.imread('constrast_image_project.png')
# -------------------------------------------------------------------------------
# # Resize an image
if img.shape[1] > 600:
    img = imutils.resize(img, width=600)
clone = img.copy()
# Convert to grayscale
gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
canny = cv2.Canny(blur, threshold1 = 40, threshold2 = 250)

# # Threshold grayscaled image to get binary image
ret, gray_threshed = cv2.threshold(canny, 78, 200,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv2.THRESH_BINARY)
#
# Smooth an image
bilateral_filtered_image = cv2.bilateralFilter(gray_threshed, 1, 1, 1)


# Find edges
t_lower = 55
t_upper = 400
edge_detected_image = cv2.Canny(bilateral_filtered_image, t_lower, t_upper )

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ---------------------------
contours, hierarchy = cv2.findContours(edge_detected_image,
    cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#Reverting the original image back to BGR so we can draw in colors
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# ---------------------------
# Displaying the results
# cv2.imshow("Original", img)
cv2.imshow('Threshed', gray_threshed)
# cv2.imshow('Smooth', bilateral_filtered_image)
# cv2.imshow("edges", edge_detected_image)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv2.imshow('Objects Detected',img)
cv2.waitKey(0)
# --------------------------------------------------------------------------------------------
# import cv2
# import numpy as np
#
# original = bilateral_filtered_image
# retval, image = cv2.threshold(original, 70, 255, cv2.THRESH_BINARY)
#
# el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# image = cv2.dilate(image, el, iterations=6)
#
# cv2.imwrite("dilated.png", image)
#
# contours, hierarchy = cv2.findContours(
#     image,
#     cv2.RETR_LIST,
#     cv2.CHAIN_APPROX_SIMPLE
# )
#
# drawing = bilateral_filtered_image
#
# centers = []
# radii = []
# for contour in contours:
#     area = cv2.contourArea(contour)
#
#     # there is one contour that contains all others, filter it out
#     if area > 500:
#         continue
#
#     br = cv2.boundingRect(contour)
#     radii.append(br[3])
#
#     m = cv2.moments(contour)
#     center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
#     centers.append(center)
#
# print("There are {} circles".format(len(centers)))
#
# radius = int(np.average(radii)) + 10
#
# for center in centers:
#     cv2.circle(drawing, center, 3, (255, 0, 0), -1)
#     cv2.circle(drawing, center, radius, (0, 255, 0), 1)
#
# cv2.imwrite("drawing.png", drawing)
# cv2.imshow("blurred", blur)
# cv2.imshow("result", drawing)
# cv2.waitKey(0)
#
#
# import numpy as np
# import cv2
# #Creating title for Streamlit app
# image = cv2.imread('constrast_image_project.png')
#
# img=image.copy()
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# blur=cv2.blur(gray,(5,5))
#
# dst=cv2.fastNlMeansDenoising(blur,None,10,7,21)
#
# _,binary=cv2.threshold(dst,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# kernel=np.ones((5,5),np.uint8)
#
# erosion=cv2.erode(binary, kernel, iterations=1)
# dilation=cv2.dilate(binary, kernel, iterations=1)
#
# if (dilation==0).sum()>1:
#     contours,_=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     for i in contours:
#         if cv2.contourArea(i)<261121.0:
#             cv2.drawContours(img,i,-1,(0,0,255),3)
#         cv2.putText(img,"defective image",(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
# else:
#     cv2.putText(img, "Good image", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
# cv2.imshow("drawing.png", img)
#
# cv2.waitKey(0)
# # cv2.destroyAllWindows()


# ----------------------------------------------------------------------------------------


# blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
# canny = cv2.Canny(blur, threshold1 = 12, threshold2 = 12)
#
# dst = cv2.fastNlMeansDenoising(canny,None,10,7,21)
#
# _, binary=cv2.threshold(dst,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# kernel=np.ones((5,5),np.uint8)
#
# erosion=cv2.erode(binary, kernel, iterations=1)
# dilation=cv2.dilate(binary, kernel, iterations=1)
#
# if (dilation==0).sum()>1:
#     contours,_=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     for i in contours:
#         if cv2.contourArea(i)<261121.0:
#             cv2.drawContours(img,i,-1,(0,0,255),3)
#         cv2.putText(img,"defective fabric",(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
# else:
#     cv2.putText(img, "Good fabric", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)