import tensorflow as tf
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

input_shape = (150, 150, 3)
batch_size = 10
train_dir = "TODO"
validation_dir = "TODO"


model = tf.keras.models.Sequential([
    Conv2D(4, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(8, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


target_size = (input_shape[0], input_shape[1])
train_generator = ImageDataGenerator(rescale=1.0/255).flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size=target_size)
validation_generator = ImageDataGenerator(rescale=1.0/255).flow_from_directory(validation_dir, batch_size=20, class_mode='binary', target_size=target_size)

history = model.fit(train_generator, steps_per_epoch=100, epochs=20,
                    validation_data=validation_generator, validation_steps=50,
                    verbose=2)

print(history)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Accuracy in training and validation')
# plt.figure()
#
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Loss in training and validation')
