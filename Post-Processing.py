import cv2
import numpy as np
import pytesseract

# def process(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)
#     binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 23)
#     kernel = np.ones((3, 3), np.uint8)
#     cleaned_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#     cv2.imshow("test", cleaned_image)
#     cv2.waitKey(0)


# image_path = 'test3.png'
# process(image_path)


def pre_process_image(img, skip_dilate=True):
    proc = cv2.GaussianBlur(img.copy(), (9, 9),0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
      kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
      proc = cv2.dilate(proc, kernel)

    proc = cv2.bitwise_not(proc)
    cv2.imshow("image", proc)
    cv2.waitKey(0)
    return proc

img = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)

processed = pre_process_image(img)