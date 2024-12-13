# import tensorflow as tf
# import numpy as np

# mnist = tf.keras.datasets.mnist

# (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
# training_data, test_data = training_data / 255, test_data / 255

# # Create a dataset of blank images and label them as 0
# num_blank_images = len(training_data) // 10  # Adjust as needed
# blank_images = np.zeros_like(training_data[:num_blank_images])
# blank_labels = np.zeros((num_blank_images,), dtype=np.uint8)

# # Combine blank images with MNIST training data
# combined_training_data = np.concatenate((training_data, blank_images), axis=0)
# combined_training_labels = np.concatenate((training_labels, blank_labels), axis=0)

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(combined_training_data, combined_training_labels, epochs=3)

# model.evaluate(test_data, test_labels)

# predictions = model.predict(test_data)
# np.set_printoptions(suppress=True)
# print(test_labels[0])
# print(predictions[0])

# model.save('digits.model')



import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('digits.h5')



# import matplotlib.pyplot as plt
# import tensorflow as tf
# import cv2 as cv
# import numpy as np

# model = tf.keras.models.load_model('digits2.model')

# for x in range(1,6):

#     image = cv.resize(cv.imread(f'{x}.png')[:,:,0], (28, 28))
#     image = np.invert(np.array([image]))
#     prediction = model.predict(image)
#     print(np.argmax(prediction))
#     plt.imshow(image[0], cmap=plt.cm.binary)
#     plt.show()



# import tensorflow as tf
# import numpy as np

# mnist = tf.keras.datasets.mnist

# (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
# training_data, test_data = training_data / 255.0, test_data / 255.0  # Normalize data

# # Generate empty images (28x28 with all zeros)
# num_empty_images = len(training_data) // 10  # Adjust the number of empty images as needed
# empty_images = np.zeros_like(training_data[:num_empty_images])

# # Create labels for empty images (0 or -, adjust as needed)
# empty_labels = np.full((num_empty_images,), 0, dtype=np.uint8)  # Use 0 for empty images

# # Combine digit and empty datasets
# combined_data = np.concatenate((training_data, empty_images), axis=0)
# combined_labels = np.concatenate((training_labels, empty_labels), axis=0)

# # Shuffle the combined dataset
# combined_indices = np.arange(len(combined_data))
# np.random.shuffle(combined_indices)
# combined_data = combined_data[combined_indices]
# combined_labels = combined_labels[combined_indices]

# # Define the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(256, activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(11, activation=tf.nn.softmax)  # 11 classes (0-9 digits + Empty)
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(combined_data, combined_labels, epochs=5)

# # Evaluate the model on test data
# model.evaluate(test_data, test_labels)


# model.save('digits3')



# import tensorflow as tf
# import numpy as np 
# import cv2
# # Load the MNIST dataset
# mnist = tf.keras.datasets.mnist
# (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
# training_data, test_data = training_data / 255, test_data / 255

# # Resize the images to 80x80 pixels
# new_image_size = (80, 80)
# training_data_resized = []
# test_data_resized = []

# for img in training_data:
#     resized_img = cv2.resize(img, new_image_size)
#     training_data_resized.append(resized_img)

# for img in test_data:
#     resized_img = cv2.resize(img, new_image_size)
#     test_data_resized.append(resized_img)

# training_data_resized = np.array(training_data_resized) / 255.0
# test_data_resized = np.array(test_data_resized) / 255.0

# # Create a dataset of blank images and label them as 0
# num_blank_images = len(training_data_resized) // 10  # Adjust as needed
# blank_images = np.zeros_like(training_data_resized[:num_blank_images])
# blank_labels = np.zeros((num_blank_images,), dtype=np.uint8)

# # Combine blank images with MNIST training data
# combined_training_data = np.concatenate((training_data_resized, blank_images), axis=0)
# combined_training_labels = np.concatenate((training_labels, blank_labels), axis=0)

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(80, 80)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(combined_training_data, combined_training_labels, epochs=3)

# model.evaluate(test_data_resized, test_labels)

# predictions = model.predict(test_data_resized)
# np.set_printoptions(suppress=True)
# print(test_labels[0])
# print(predictions[0])

# model.save('digits_80x80.model')


# import tensorflow as tf
# import numpy as np

# mnist = tf.keras.datasets.mnist

# (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
# training_data, test_data = training_data / 255, test_data / 255

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(training_data, training_labels, epochs=5)

# model.evaluate(test_data, test_labels)

# predictions = model.predict(test_data)
# np.set_printoptions(suppress=True)
# print(test_labels[0])
# print(predictions[0])

# model.save('digits.model')


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
# from keras.datasets import mnist
# import matplotlib.pyplot as plt

# # Load the MNIST dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# num_blank_images = len(X_train) // 10  # Adjust as needed
# blank_images = np.zeros_like(X_train[:num_blank_images])
# blank_labels = np.zeros((num_blank_images,), dtype=np.uint8)


# combined_training_data = np.concatenate((X_train, blank_images), axis=0)
# combined_training_labels = np.concatenate((y_train, blank_labels), axis=0)


# # Preprocessing the data
# X_train = X_train.reshape(X_train.shape + (1,))
# X_test = X_test.reshape(X_test.shape + (1, ))
# X_train = X_train / 255.
# X_test = X_test / 255.
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)

# # Create and compile the model
# model = tf.keras.Sequential([
#     layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(28,  28,  1)),
#     layers.Conv2D(10,  3, activation="relu"),
#     layers.MaxPool2D(),
#     layers.Conv2D(10,  3, activation="relu"),
#     layers.Conv2D(10,  3, activation="relu"),
#     layers.MaxPool2D(),
#     layers.Flatten(),
#     layers.Dense(10, activation="softmax")  # 10 classes for digits 0-9
# ])

# model.summary()

# model.compile(loss="sparse_categorical_crossentropy", 
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=["accuracy"])

# # Train the model
# model.fit(combined_training_data, combined_training_labels, epochs=3)

# # Evaluate the model
# # Save the model
# model.save("digit-recognizer3.h5")

# # Randomly select and display an image from the test set
# random_image = np.random.randint(0, len(X_test))
# predicted_label = model.predict(X_test[random_image:random_image+1]).argmax()
# plt.imshow(X_test[random_image].reshape(28, 28), cmap="gray")
# plt.title(f"Predicted: {predicted_label}")
# plt.axis(False)
# plt.show()
