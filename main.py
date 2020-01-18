import os
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

IMG_HEIGHT = 150
IMG_WIDTH = 150
_epochs = 200
batch_size = 16

def main():
  base_dir = '/home/ubuntu/project/ojt'

  train_dir = os.path.join(base_dir, 'train')
  validation_dir = os.path.join(base_dir, 'validation')
  test_dir = os.path.join(base_dir, 'test')

 # model = models.Sequential()
 # model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
 # model.add(layers.Conv2D(16, (3, 3), activation='relu'))
 # model.add(layers.MaxPooling2D((2, 2)))
 # model.add(layers.Conv2D(32, (3, 3), activation='relu'))
 # model.add(layers.MaxPooling2D((2, 2)))
 # model.add(layers.Flatten())
 # model.add(layers.Dense(512, activation='relu'))
 # model.add(layers.Dense(256, activation='relu'))
 # model.add(layers.Dense(1, activation='sigmoid'))

#  model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
  
  model = models.Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
  ])
  
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

  train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45, horizontal_flip=True, zoom_range=0.)
  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size, class_mode='binary')

  validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size, class_mode='binary')

#  checkpoint_cb = ModelCheckpoint("snapshot/{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)

 
  history = model.fit_generator(train_generator, steps_per_epoch=15, epochs=_epochs, validation_data=validation_generator, validation_steps=50)#, callbacks=[checkpoint_cb])


  model.save(os.path.join(base_dir, 'output/model.h5'))

  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(1, len(acc) + 1)

  plt.figure(figsize=(8,8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, acc, label='Training Acc')
  plt.plot(epochs, val_acc, label='Validation Acc')
  plt.title('Training and validation accuracy')
  plt.legend(loc='lower right')

  plt.subplot(1, 2, 2)
  plt.plot(epochs, loss, label='Training loss')
  plt.plot(epochs, val_loss, label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend(loc='upper right')

  plt.savefig('result.png')

if __name__ == "__main__":
	main()
