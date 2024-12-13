# from keras.datasets import mnist

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(f"We have {len(X_train)} images in the training set and {len(X_test)} images in the test set.")

# X_train[0].shape

# import matplotlib.pyplot as plt
# plt.imshow(X_train[0])

# plt.figure(figsize=(3, 3))
# plt.imshow(X_train[0], cmap="gray")
# plt.title(y_train[0])
# plt.axis(False);

# import random
# random_image = random.randint(0,  len(X_train))

# plt.figure(figsize=(3, 3))
# plt.imshow(X_train[random_image], cmap="gray")

# plt.title(y_train[random_image])
# plt.axis(False);

# X_train.shape

# X_train = X_train.reshape(X_train.shape + (1,))
# X_test = X_test.reshape(X_test.shape + (1, ))

# X_train.shape

# X_train = X_train / 255.
# X_test = X_test / 255.

# import numpy as np
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)


# import tensorflow as tf
# from tensorflow.keras import layers

# model = tf.keras.Sequential([
# 	layers.Conv2D(filters=10,
# 				kernel_size=3, 
# 				activation="relu", 
# 				input_shape=(28,  28,  1)),
# 	layers.Conv2D(10,  3, activation="relu"),
# 	layers.MaxPool2D(),
# 	layers.Conv2D(10,  3, activation="relu"),
# 	layers.Conv2D(10,  3, activation="relu"),
# 	layers.MaxPool2D(),
# 	layers.Flatten(),
# 	layers.Dense(10, activation="softmax")
# ])

# model.summary()


# model.compile(loss="sparse_categorical_crossentropy", 
# 			optimizer=tf.keras.optimizers.Adam(),
# 			metrics=["accuracy"])


# model.fit(X_train, y_train, epochs=5)


# model.evaluate(X_test, y_test)


# model.save("digit-recognizer.h5")




















import tkinter as tk
from PIL import Image, ImageGrab
import numpy as np
import win32gui
import tensorflow as tf



model = tf.keras.models.load_model('digit-recognizer.h5')
def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    img = 1 - img
    #predicting
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.is_drawing = False  # Boolean variable to track if left mouse button is pressed
        # Creating elements
        self.canvas = tk.Canvas(self, width=400, height=400, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.btn_classify = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.clear_button = tk.Button(self, text="Clear", command=self.clear_all)
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.btn_classify.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button.grid(row=1, column=0, pady=2)
        self.canvas.bind("<Button-1>", self.start_drawing)  # Bind left mouse button press event
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)  # Bind left mouse button release event
        self.canvas.bind("<B1-Motion>", self.draw_lines)  # Bind mouse motion event

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def start_drawing(self, event):
        self.is_drawing = True
        self.x = event.x
        self.y = event.y

    def stop_drawing(self, event):
        self.is_drawing = False

    def draw_lines(self, event):
        if self.is_drawing:
            x1, y1 = (self.x, self.y)
            x2, y2 = (event.x, event.y)
            self.canvas.create_line(x1, y1, x2, y2, fill='black', width=8)
            self.x = x2
            self.y = y2

# Rest of your code for loading the model and predicting digits

app = App()
app.mainloop()