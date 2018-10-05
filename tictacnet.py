import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Cool ML here:

df = pd.read_csv('tictactoe-data.csv')

X = df.iloc[:, list(range(18)) + [-2]]
#print(X.head)

# we are doing classification (different from regression)
# m variables are moves
# 0 = don't place a mark here, 1 = place a mark here
target = df.iloc[:, list(range(18, 27))]

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=X.shape[1]))
model.add(tf.keras.layers.Dropout(0.3))  # Avoid overfitting. automatically turns off when not training
model.add(tf.keras.layers.Dense(64, activation='relu'))  # don't have to specify number of inputs
model.add(tf.keras.layers.Dropout(0.3))  # Avoid overfitting
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))  # Avoid overfitting
model.add(tf.keras.layers.Dense(target.shape[1], activation ='softmax')) #output, target.shape[1] = 9
# Dense = fully connected layer. 128 neurons in the layer


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # accuracy how similar is the aoutput to what I want

# print(model.summary())
# foo

# epoch = how many times I want to iterate
model.fit(X_train,
          y_train,
          epochs=100,
          batch_size=12,
          validation_data=[X_test, y_test])

print('accuracy:', model.evaluate(X_test, y_test))

# save the model
model.save('tictacNET.h5')
