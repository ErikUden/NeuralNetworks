import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt
#Import of TensorFLow, the library Keras, NumPy and Matplot/Pyplot for maths



data = keras.datasets.fashion_mnist
#The (learning) Data is grabbed from the official TensorFlow/Keras Dataset

(train_images, train_labels), (test_images, test_labels) = data.load_data()
#The learning images are devided into train images and test images for obvious reasons. (To test whether or not the AI is actually smart or just remembering each picture individually, thereby never work on a real-life setting)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#The official class names from the Keras Lib


train_images=train_images/255.0
test_images=test_images/255.0
#All the 3 greyscale "dimensions" which represent RGB are devided by the value 255 (the max value of RGB coloring for each red, green or blue) in order to create a digit always above 0 and below 1 (0<RGB<1) to make it "understandable" for the layers and output.



model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")])
#Layers wíth help of the keras library are defined. 




model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10)
#Too complicated to explain, Adam and Crossentropy are pretty common and popular, google it and go through that!
#Change the epoch value on your behalf

prediction = model.predict(test_images)
#The AI is tested to predict images it has never seen before.



for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction " + class_names[np.argmax(prediction[i])])
	plt.show()

