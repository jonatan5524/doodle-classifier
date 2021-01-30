import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from dataset import load_data, classes
import matplotlib.pyplot as plt
import os

class Model:

  def __init__(self, class_num = 4, input_shape = (28, 28, 1)):
    self.checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    self.model = self.build_model(class_num, input_shape)

    training_dataset, test_dataset = load_data()
    self.training_labels, self.training_images = Model.encode_dataset(training_dataset)
    self.test_labels, self.test_images = Model.encode_dataset(test_dataset)

    self.batch_size = 25
    self.epochs = 100

  def load(self):
    latest = tf.train.latest_checkpoint(self.checkpoint_dir)
    self.model.load_weights(latest)

  def build_model(self,class_num, input_shape):
    model = models.Sequential()

    model.add(
      layers.Conv2D(
        filters = 28,
        kernel_size = (3, 3),
        input_shape = input_shape,
        activation='relu'
      )
    )
    model.add(
      layers.MaxPooling2D(
        pool_size = (2, 2),
      )
    )
    model.add(
      layers.Conv2D(
        filters = 56,
        kernel_size = (3, 3),
        activation='relu'
      )
    )
    model.add(
      layers.MaxPooling2D(
        pool_size = (2, 2),
      )
    )
    model.add(
      layers.Conv2D(
        filters = 56,
        kernel_size = (3, 3),
        activation='relu'
      )
    )
    model.add(
      layers.Flatten()
    )
    model.add(
      layers.Dense(
        units = class_num,
        activation = 'softmax'
      )
    )

    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )

    return model

  def train(self):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=self.checkpoint_path, 
      verbose=1, 
      save_weights_only=True,
      save_freq=5*self.batch_size
    )

    return self.model.fit(
      x = self.training_images,
      y = self.training_labels,
      batch_size = self.batch_size,
      epochs = self.epochs,
      shuffle = True,
      validation_data = (self.test_images, self.test_labels),
      callbacks=[cp_callback]
    )

  def test(self):
    return self.model.evaluate(
      x = self.test_images,
      y = self.test_labels,
      batch_size = self.batch_size,
      verbose = 2
    )

  def predict(self, image):
    image = np.array([np.array(image).astype(np.float).reshape((28, 28, 1))])
    prediction = self.model.predict(image)
    
    return list(classes.keys())[np.argmax(prediction)]

  @staticmethod
  def encode_dataset(dataset):
    labels, images = list(map(list, zip(*dataset)))
    
    encoded_labels = np.array(list(map(lambda label: classes[label], labels)))
    images = np.array(images).reshape(len(images), 28, 28, 1)

    return encoded_labels, images

  def progress_graph(self, history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(self.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

  def print(self):
    self.model.summary()

  @staticmethod
  def display_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.astype("uint8"))
    plt.axis("off")
    plt.show()