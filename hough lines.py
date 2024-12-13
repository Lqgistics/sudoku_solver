import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageGrab
import win32gui


def remove_sudoku_grid(image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 23)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    lines = cv2.HoughLinesP(cleaned_image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    inverted_image = cv2.bitwise_not(image)

    return inverted_image

def pre_process_image(cell, skip_dilate=False):
    proc = cv2.bitwise_not(cell)
    cv2.imshow("cell", proc)
    cv2.waitKey(0)
    proc = cv2.medianBlur(proc, 5)
    cv2.imshow("cell", proc)
    cv2.waitKey(0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    cv2.imshow("cell", proc)
    cv2.waitKey(0)
    proc = cv2.bitwise_not(proc, proc)
    cv2.imshow("cell", proc)
    cv2.waitKey(0)
    if not skip_dilate:
      kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
      proc = cv2.dilate(proc, kernel)
    cv2.imshow("cell", proc)
    cv2.waitKey(0)
    proc = cv2.bitwise_not(proc)
    cv2.imshow("cell", proc)
    cv2.waitKey(0)
    return proc


def predict_digit(cell):
    cell = pre_process_image(cell)
    img = cv2.resize(cell, (28,28))
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    img = 1 - img

    res = model.predict([img])[0]

    digit = np.argmax(res)

    return digit

def visualize_cells(image_path):
    result_image = remove_sudoku_grid(image_path)
    height, width = result_image.shape
    cell_width = width // 9
    cell_height = height // 9
    
    cv2.namedWindow("Sudoku Image with Cell Text", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sudoku Image with Cell Text", 500, 500)

    for i in range(9):
        for j in range(9):
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = (j + 1) * cell_width
            y2 = (i + 1) * cell_height
            
            cell = result_image[y1:y2, x1:x2]

            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cell_text = predict_digit(cell)

            cv2.putText(result_image, str(cell_text), (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("Sudoku Image with Cell Text", result_image)
    cv2.waitKey(0)



model = tf.keras.models.load_model('digit-recognizer.h5')

image_path = 'board.png'
visualize_cells(image_path)
