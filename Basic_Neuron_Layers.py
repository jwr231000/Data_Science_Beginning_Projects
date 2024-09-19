import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist
(X,y), (X_test,y_test) = mnist.load_data()

X_valid, X_train = X[:5000] / 255.0 , X[5000:] / 255.0
y_valid, y_train = y[:5000], y[5000:]

class_names = ['0','1','2','3','4','5','6','7','8','9']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='softmax')])

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

model.fit(X_train,y_train, epochs=30, validation_data=(X_valid,y_valid))

model.evaluate(X_train,y_train)