import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the digits dataset
digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))

# Calculate moments for each image
moments_list = []
for image in digits.images:
    img = np.array(image, dtype=np.uint8)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # Check if contours exist
        cnt = max(contours, key=cv2.contourArea)
        m = cv2.moments(cnt)
        moments_list.append([m['m00'], m['m10'], m['m01'], m['m20'], m['m11'], m['m02'], m['m30'], m['m21'], m['m12'], m['m03']])
    else:
        moments_list.append([0] * 10)  # If no contours found, append zeros

moments_array = np.array(moments_list)


merged_df = pd.concat([pd.DataFrame(moments_array), pd.DataFrame(data)], axis=1)
df_data= pd.DataFrame(merged_df)
df_data = merged_df.astype('float32')


# Split the data
X_train, X_test, y_train, y_test = train_test_split(df_data, digits.target, test_size=0.2, shuffle=False)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(df_data.shape[1],)), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:',test_acc)# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)