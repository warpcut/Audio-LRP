import warnings
warnings.simplefilter('ignore')
import imp
import matplotlib.pyplot as plot
import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils
from keras import regularizers, optimizers
import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
from memory_profiler import memory_usage
import pandas as pd
from glob import glob
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path
from keras_preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def append_ext(fn):
    result = os.path.splitext(fn)[0]
    return result +".png"

def remove_ext(fn):
    return os.path.splitext(fn)[0]

traindf=pd.read_csv('./urban/UrbanSound8K.csv',dtype=str)
testdf=pd.read_csv('./urban/UrbanSound8K.csv',dtype=str)
traindf["slice_file_name"]=traindf["slice_file_name"].apply(append_ext)
testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./urban/train/",
    x_col="slice_file_name",
    y_col="class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./urban/train/",
    x_col="slice_file_name",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

input_shape=(64,64,3)
'''
# Create model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu", name='dense_512'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax", name='dense_out'),
])

optimizer = keras.optimizers.Adam(lr=0.0001)
#optimizer = keras.optimizers.rmsprop(lr=0.0005, decay=1e-6)
model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator = train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25
)

scores = model.evaluate_generator(generator = valid_generator, steps=STEP_SIZE_VALID)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
model.save('./models/model_25_mat_adam.h5')
print("Model saved")

'''
# Recreate the exact same model, including its weights and the optimizer
model = keras.models.load_model('./models/model_30_mat_adam.h5')
# Show the model architecture
model.summary()


test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
  dataframe=testdf,
  directory="./urban/test/siren",
  x_col="slice_file_name",
  y_col=None,
  batch_size=32,
  seed=42,
  shuffle=False,
  class_mode=None,
  target_size=(64,64))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames

model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
# model_wo_sm = model
test_generator.reset()
for counter in range(0, 1000):
  image = next(test_generator)
  plt.imshow(image[0], cmap="seismic", clim=(-1, 1))
  plt.axis('off')
  filename = './images/siren/' + remove_ext(filenames[counter]) + "_" + predictions[counter] + '.png'
  plt.savefig(filename)
  '''
  #Gradient
  analyzer = innvestigate.create_analyzer("gradient", model_wo_sm)
  a = analyzer.analyze(image)

  a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
  a /= np.max(np.abs(a))
  # Plot
  plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
  plt.axis('off')
  filename = './images/siren/' + remove_ext(filenames[counter]) + '_gradient.png'
  plt.savefig(filename)
  '''
  #LRP epsilon
  analyzer = innvestigate.create_analyzer("lrp.epsilon",model_wo_sm)
  a = analyzer.analyze(image)
  a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
  a /= np.max(np.abs(a))

  plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
  plt.axis('off')
  filename = './images/siren/' + remove_ext(filenames[counter]) + '_lrp_epsilon.png'
  plt.savefig(filename)

  # LRP flat A
  analyzer = innvestigate.create_analyzer("lrp.sequential_preset_a_flat",model_wo_sm)
  a = analyzer.analyze(image)
  a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
  a /= np.max(np.abs(a))

  plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
  plt.axis('off')
  filename = './images/siren/' + remove_ext(filenames[counter]) + '_lrp_flat.png'
  plt.savefig(filename)
  plot.close()
  plot.close('all')
