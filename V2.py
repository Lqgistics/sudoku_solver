import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import pytesseract

im = cv2.imread("board.png")
im = cv2.resize(im, (900, 900))

out = np.zeros((9, 9), dtype=np.uint8)

model = tf.keras.models.load_model('digit-recognizer2.h5')

for x in range(9):
    for y in range(9):

        cell = im[10 + x * 100:(x + 1) * 100 - 10, 10 + y * 100:(y + 1) * 100 - 10, :]

        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        print(cell.shape)
        cell = cv2.resize(cell, (28, 28))

        image = np.invert(np.array([cell]))

        prediction = model.predict(image)

        predicted_digit = np.argmax(prediction)
        
        print(predicted_digit)

        if predicted_digit:
            out[x, y] = predicted_digit

print(out)


# import cv2
# from imutils import contours
# import numpy as np
# import tensorflow as tf

# # Load image, grayscale, and adaptive threshold
# image = cv2.imread("board.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)

# # Filter out all numbers and noise to isolate only boxes
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 1000:
#         cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

# # Fix horizontal and vertical lines
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

# # Sort by top to bottom and each row by left to right
# invert = 255 - thresh
# cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

# sudoku_rows = []
# row = []
# for (i, c) in enumerate(cnts, 1):
#     area = cv2.contourArea(c)
#     if area < 50000:
#         row.append(c)
#         if i % 9 == 0:  
#             (cnts, _) = contours.sort_contours(row, method="left-to-right")
#             sudoku_rows.append(cnts)
#             row = []

# # Iterate through each box
# for row in sudoku_rows:
#     for c in row:
        
#         mask = np.zeros(image.shape, dtype=np.uint8)
#         cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
#         result = cv2.bitwise_and(image, mask)
#         result[mask == 0] = 255

#         cv2.imshow('Cell', result)
#         cv2.waitKey(175)

# def detection(warped):
#     out = np.zeros((9, 9), dtype=np.uint8)
#     for x in range(9):
#         for y in range(9):
#             cell = warped[10 + x * 100:(x + 1) * 100 - 10, 10 + y * 100:(y + 1) * 100 - 10]

#             if cell.size == 0:
#                 continue

#             cell = cv2.resize(cell, (28, 28))
#             image = np.invert(np.array([cell]))

#             prediction = model.predict(image)
#             predicted_digit = np.argmax(prediction)

#             if predicted_digit:
#                 out[x, y] = predicted_digit

#     print(out)

# model = tf.keras.models.load_model('digit-recognizer2.h5')
# out = np.zeros((9, 9), dtype=np.uint8)


# cv2.waitKey()
