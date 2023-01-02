from tensorflow import keras
from PIL import Image
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_test = x_test / 255.0
x_train = x_train / 255.0

y_test = tf.keras.utils.to_categorical(y_test, 10)
y_train = tf.keras.utils.to_categorical(y_train, 10)

image = Image.open("dog.jpg")
image = image.resize((32,32))

numpydata = np.asarray(image)
model = keras.models.load_model('my_model')

arr = np.array([numpydata])
print(arr.shape)
arr = arr / 255.0
result = model.predict(arr)

prediction = np.argmax(result[0])

array = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(f"This is an image of {array[prediction]}")


    





