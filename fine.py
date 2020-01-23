import os
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras import models, regularizers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

IMG_HEIGHT = 150
IMG_WIDTH = 150
_epochs = 200
batch_size = 32

def main():

  base_model = VGG16(weights='imagenet', include_top=False, input_tensor=(150, 150, 3)))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(256, activation='relu')(x)
  prediction = Dense(3, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=prediction)

  for layer in base_model.layers[:15]:
      layer.trainable = False

  model.compile(optimzer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

  base_dir = '/home/ubuntu/project/ojt'

  train_dir = os.path.join(base_dir, 'train')
  validation_dir = os.path.join(base_dir, 'validation')

  labels = []
  with open('./labels.txt', 'r') as f:
    for line in f:
      labels.append(line.rstrip())
  print(labels)

  num_train = 0
  num_validation = 0

  for label in labels:
    num_train = num_train + len(os.listdir(os.path.join(train_dir, label)))
    num_validation = num_validation + len(os.listdir(os.path.join(validation_dir, label)))

  print('train images : ' + str(num_train))
  print('validation images : ' + str(num_validation))

  train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, horizontal_flip=True, zoom_range=0.25, width_shift_range=.10, height_shift_range=.10)
  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size, class_mode='categorical')

  validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size, class_mode='categorical')

#  checkpoint_cb = ModelCheckpoint("snapshot/{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)


  history = model.fit_generator(train_generator, steps_per_epoch=num_train, epochs=_epochs, validation_data=validation_generator, validation_steps=num_validation)#, callbacks=[checkpoint_cb])


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
