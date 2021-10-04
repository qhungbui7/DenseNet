import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, ZeroPadding2D, Input, MaxPool2D, Concatenate, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import os
import matplotlib.pyplot as plt


TRAINING_DIR = '/content/train'
VALIDATION_DIR = '/content/valid'
TESTING_DIR = '/content/test'
FOLDER_DIR = ' /content/drive/MyDrive/DeepLearning/DenseNet/'

BATCH_SIZE = 32
INPUT_SHAPE = (224, 224)
SEED = 777

training_gen = ImageDataGenerator(rescale = 1./255 # ,
                                  # width_shift_range= 0.2, 
                                  # height_shift_range= 0.2, 
                                  # zoom_range=0.2, 
                                  # vertical_flip = True,
                                  # rotation_range = 40
                                  )
training_set = training_gen.flow_from_directory(TRAINING_DIR,
                                                batch_size = BATCH_SIZE,
                                                target_size = INPUT_SHAPE, 
                                                class_mode = 'categorical'
                                                )
validation_gen = ImageDataGenerator(rescale = 1./255)
validation_set = validation_gen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size = BATCH_SIZE,
                                                    target_size = INPUT_SHAPE, 
                                                    class_mode = 'categorical'
                                                    )
testing_gen = ImageDataGenerator(rescale = 1./255)
testing_set = testing_gen.flow_from_directory(TESTING_DIR,
                                              batch_size = BATCH_SIZE,
                                              target_size = INPUT_SHAPE, 
                                              class_mode = 'categorical'
                                              )

class SaveModelCertainEpochs(tf.keras.callbacks.Callback) : 
  def on_epoch_end(self, epoch, logs={}) :
    if epoch % 5 == 0 and epoch > 5:      
      print('Saved in' + FOLDER_DIR + 'model_{}_densenet.hd5 sucessfully !'.format(epoch))
      self.model.save(FOLDER_DIR + 'model_{}_densenet.hd5'.format(epoch))
callback_1 = SaveModelCertainEpochs()      
class LearningRateDecay(tf.keras.callbacks.Callback) : # mark
  def on_epoch_begin(self, epoch, logs = {}) : 
    if epoch < 20:
      return 
    else:
      self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * tf.math.exp(-0.1)
callback_2 = LearningRateDecay()
class EarlyStopping(tf.keras.callbacks.Callback) : 
  def on_epoch_end(self, epoch, logs = {}) : 
    if logs['val_accuracy'] > 0.90 : 
      self.model.stop_training = True 
      print('Early Stopping !, learning rate now is {}'.format(self.model.optimizer.learning_rate)) 
callback_3 = EarlyStopping()
class SaveBestModel(tf.keras.callbacks.Callback) : 
  def __init__(self, save_best_metric='val_loss') : 
    self.save_best_metric = save_best_metric
    self.best = float('inf')
  def on_epoch_end(self, epoch, logs={}) : 
    metric_val = logs[self.save_best_metric]
    if metric_val < self.best : 
      self.best = metric_val
      self.best_weights = self.model.get_weights()
callback_4 = SaveBestModel()

class ConvBlock() : 
  def __init__(self, filters) :
    super().__init__() 
    self.filters = filters
  def __call__(self, x):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(self.filters, 1)(x)
    x = Conv2D(self.filters, 3, padding = 'same')(x)
    return x

class DenseBlock() : 
  def __init__(self, filters) : 
    super().__init__() 
    self.conv_block = ConvBlock(filters)
    self.cont = Concatenate(axis=-1)
  def __call__(self, x, num_conv_block) :
    inputs = [x] 
    for i in range(num_conv_block) :
      x = self.cont(inputs)
      x = self.conv_block(x)
      inputs.append(x)
    
    x = self.cont(inputs)
    return x

class TransistionBlock() : 
  def __init__(self, filters) : 
    super().__init__()
    self.bn = BatchNormalization()
    self.activation = ReLU()
    self.conv_1 = Conv2D(filters, 1)
    self.avgpooling = AveragePooling2D(pool_size = 2, strides = 2)
  def __call__(self, x):
    x = self.bn(x)
    x = self.activation(x)
    x = self.conv_1(x)
    x = self.avgpooling(x)
    return x

class DenseNet(tf.keras.Model) : 
  def __init__(self, grow_rate, base_dims, num_classes, input_shape) : 
    super().__init__()
    self.grow_rate = grow_rate
    self.base_dims = base_dims
    self.inputs = Input(shape = input_shape)
    self.conv_7 = Conv2D(base_dims, kernel_size = 7, strides = 2, padding = 'same')
    self.bn = BatchNormalization()
    self.activation = ReLU()
    self.max_pooling = MaxPooling2D(pool_size=3, strides=2, padding = 'same')
    self.glob_avg_pool = GlobalAveragePooling2D()
    self.classifier = Dense(num_classes, activation = 'softmax')

  def __call__(self) : 
    num_conv_block = [6, 12, 24, 16]
    inputs = self.inputs
    x = self.conv_7(inputs)
    x = self.bn(x)
    x = self.activation(x)
    x = self.max_pooling(x)
    for i in range (0, 4) : 
      dims = self.base_dims + i * self.grow_rate
      x = DenseBlock(dims)(x, num_conv_block[i])
      if (i != 3) : 
        x = TransistionBlock(dims)(x)
        
    x = self.glob_avg_pool(x)
    classifier = self.classifier(x)

    return tf.keras.Model(inputs = inputs, outputs = classifier)

EPOCHS = 30
GROW_RATE = 32
BASE_FILTERS = 64

model = DenseNet(32, 64, 300, (224, 224, 3))()
lrs = [0.001, 0.0001]
for lr in lrs : 
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'categorical_crossentropy', metrics=['accuracy'])
  print(model.summary())
  history = model.fit(training_set, 
                      batch_size = BATCH_SIZE, 
                      epochs = EPOCHS, 
                      validation_data = validation_set, 
                      verbose = 1, 
                      callbacks = [callback_1, callback_2, callback_3, callback_4])