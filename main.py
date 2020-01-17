import os
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

def main():
  base_dir = '/home/ubuntu/project/ojt'

  train_dir = os.path.join(base_dir, 'train')
  validation_dir = os.path.join(base_dir, 'validation')
  test_dir = os.path.join(base_dir, 'test')

  model = models.Sequential()
  model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
  model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))


  model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

  train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45)
  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(train_dir, target_size=(256, 256), batch_size=20, class_mode='binary')

  validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(256, 256), batch_size=20, class_mode='binary')

  history = model.fit_generator(train_generator, steps_per_epoch=1000, epochs=20, validation_data=validation_generator, validation_steps=50)


  model.save(os.path.join(base_dir, 'output/model.h5'))

  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(1, len(acc) + 1)

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.savefig('result_1.png')
  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.savefig('result_2.png')

if __name__ == "__main__":
	main()
