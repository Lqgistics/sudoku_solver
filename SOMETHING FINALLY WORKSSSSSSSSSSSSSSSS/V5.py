import cv2
import numpy as np
import tensorflow as tf
import operator
import tkinter as tk
from PIL import Image, ImageTk
import pymongo
from pymongo import MongoClient

from tkinter import *

vid = cv2.VideoCapture(0)

width, height = 800, 600

vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

app = Tk()

# Function to close the camera, take a snapshot, and quit the application
def close_camera():
    _, frame = vid.read()
    cv2.imwrite("irl.jpg", frame)
    app.quit()

label_widget = Label(app)
label_widget.pack()

# Function to open the camera, continuously capture frames, and display them in a Tkinter label
def open_camera():
    button1.pack_forget()
    button2.pack()

    _, frame = vid.read()

    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    captured_image = Image.fromarray(opencv_image)

    photo_image = ImageTk.PhotoImage(image=captured_image)

    label_widget.photo_image = photo_image

    label_widget.configure(image=photo_image)

    # Repeat the same process after every 10 milliseconds
    label_widget.after(10, open_camera)

button1 = Button(app, text="Open Camera", command=open_camera)
button1.pack()

button2 = Button(app, text="Capture Image", command=close_camera)

app.mainloop()

# Function to preprocess the image for Sudoku recognition
def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9),0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc

# Function to find the corners of a Sudoku grid
def findCorners(processed):
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

# Function to display points on an image
def display_points(in_img, points, radius=25, colour=(0, 0, 255)):
    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    return img

def distance_between(p1, p2):
    # Calculate the distance between two points in 2D space
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def display_rects(in_img, rects, colour=255):
    # Draw rectangles on an image given their coordinates
    img = in_img.copy()
    for rect in rects:
        cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    return img

def crop_and_warp(img, crop_rect):
    # Crop and warp the input image to a square shape
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def remove_sudoku_grid(cropped):
    # Remove the grid lines from a cropped sudoku image
    blur = cv2.GaussianBlur(cropped, (9, 9), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 23)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    lines = cv2.HoughLinesP(cleaned_image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cropped, (x1, y1), (x2, y2), (255, 255, 255), 2)
                                  
    return cropped

def visualize_cells(result_image):
    # Detect digits in each cell of the sudoku grid
    height, width = result_image.shape
    cell_width = width // 9
    cell_height = height // 9
    sudoku_array = np.zeros((9, 9), dtype=int)

    for i in range(9):
        for j in range(9):
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = (j + 1) * cell_width
            y2 = (i + 1) * cell_height
            cell = result_image[y1:y2, x1:x2]
            unique_colours = np.unique(cell)
            num_colours = len(unique_colours)
            img = np.array(cell)
            img = cv2.resize(img, (48, 48))
            img = img.reshape(-1, 48, 48, 1)
            if num_colours > 128:
                prediction = model.predict(img)
                predicted_digit = np.argmax(prediction)
            else:
                predicted_digit = 0
            sudoku_array[i, j] = predicted_digit
    
    return sudoku_array

def solve(grid, row, col, num):
    # Solve the sudoku grid recursively using backtracking
    for x in range(9):
        if grid[row][x] == num:
            return False
             
    for x in range(9):
        if grid[x][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def Suduko(grid, row, col):
    # Recursive function to solve the sudoku grid
    if (row == 9 - 1 and col == 9):
        return True
    if col == 9:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Suduko(grid, row, col + 1)
    for num in range(1, 9 + 1, 1): 
        if solve(grid, row, col, num):
            grid[row][col] = num
            if Suduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

# Load the OCR model
model = tf.keras.models.load_model('model-OCR.h5')

# Function to process and display the Sudoku puzzle
def process_and_display():
    global grid, answer_button, hints_button
    # Hide the start button
    start_button.pack_forget()
    # Read the Sudoku image
    img = cv2.imread('irl.jpg', cv2.IMREAD_GRAYSCALE)
    # Preprocess the image
    processed = pre_process_image(img)
    # Find corners of the Sudoku puzzle
    corners = findCorners(processed)
    # Crop and warp the Sudoku puzzle
    cropped = crop_and_warp(img, corners)
    # Remove the Sudoku grid lines
    result_image = remove_sudoku_grid(cropped)
    # Visualize the cells of the Sudoku puzzle
    grid = visualize_cells(result_image)

    # Enable the answer and hints buttons
    answer_button.config(state=tk.NORMAL)
    answer_frame.pack()

    hints_button.config(state=tk.NORMAL)
    hints_frame.pack()

# Function to display the solved Sudoku puzzle
def show_answers():
    global grid
    # Solve the Sudoku puzzle
    if Suduko(grid, 0, 0):
        # Create a Tkinter window
        root = tk.Tk()
        root.title("Sudoku Grid")
        root.geometry("400x400") 
        
        frame = tk.Frame(root)
        frame.pack()

        # Display the solved Sudoku puzzle
        for i in range(9):
            for j in range(9):
                digit = grid[i][j]
                label = tk.Label(frame, text=str(digit) if digit != 0 else "", width=3, relief="ridge", font=("Arial", 12))
                label.grid(row=i, column=j, padx=2, pady=2)

        root.mainloop()
    else:
        print("Solution does not exist")

# Function to reveal digits in the Sudoku puzzle as hints
def reveal_digit(i, j):
    global grid, buttons_grid
    digit = grid[i][j]
    buttons_grid[i][j].config(text=str(digit) if digit != 0 else "")

# Function to display hints for solving the Sudoku puzzle
def show_hints():
    global grid, buttons_grid
    # Solve the Sudoku puzzle
    if Suduko(grid, 0, 0):
        # Create a Tkinter window
        root = tk.Tk()
        root.title("Sudoku Grid")
        root.geometry("400x400") 
        
        frame = tk.Frame(root)
        frame.pack()

        buttons_grid = []

        # Display buttons for each cell in the Sudoku puzzle
        for i in range(9):
            row_buttons = []
            for j in range(9):
                button = tk.Button(frame, text="", width=3, relief="raised", font=("Arial", 12),
                                   command=lambda i=i, j=j: reveal_digit(i, j))
                button.grid(row=i, column=j, padx=2, pady=2)
                row_buttons.append(button)
            buttons_grid.append(row_buttons)

        root.mainloop()
    else:
        print("Solution does not exist")

# Function to create the graphical user interface (GUI)
def GUI():
    global start_frame, start_button, answer_frame, answer_button, hints_frame, hints_button
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Sudoku Solver")
    root.geometry("400x400") 

    # Create the frame for the start button
    start_frame = tk.Frame(root)
    start_frame.pack()

    # Create the start button
    start_button = tk.Button(start_frame, text="Start", command=process_and_display, width=10, height=2, font=("Arial", 12), bg="#7092BE", fg="white")
    start_button.pack()

    # Create the frame for the answer button
    answer_frame = tk.Frame(root)
    answer_button = tk.Button(answer_frame, text="Show Answers", command=show_answers, state=tk.DISABLED, width=15, height=2, font=("Arial", 12), bg="#7092BE", fg="white")
    answer_button.pack()

    # Create the frame for the hints button
    hints_frame = tk.Frame(root)
    hints_button = tk.Button(answer_frame, text="Show Hints", command=show_hints, state=tk.DISABLED, width=15, height=2, font=("Arial", 12), bg="#7092BE", fg="white")
    hints_button.pack()

    root.mainloop()

# Execute the GUI function if the script is run as the main program
if __name__ == "__main__":
    GUI()
