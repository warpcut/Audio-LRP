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
import pandas as pd
import pylab
import matplotlib.pyplot as plt
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

# Load model, including its weights and the optimizer
model = keras.models.load_model('./models/model_30_mat_adam.h5')
# Show the model architecture
model.summary()

testdf=pd.read_csv('./urban/UrbanSound8K.csv',dtype=str)
testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)

#cat_size = 929
#cat = "siren"
cat_size = 429
cat = "car_horn"
#cat_size = 1000
#cat = "drilling"
#cat_size = 374
#cat = "gun_shot"
#cat_size = 1000
#cat = "street_music"
#cat_size = 1000
#cat = "jackhammer"
#cat_size = 1000
#cat = "air_conditioner"
#cat_size = 1000
#cat = "children_playing"
#cat_size = 1000
#cat = "dog_bark"
#cat_size = 1000
#cat = "engine_idling"

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
  dataframe=testdf,
  directory="./urban/test/" + cat,
  x_col="slice_file_name",
  y_col=None,
  batch_size=32,
  seed=42,
  shuffle=False,
  class_mode=None,
  target_size=(64,64))
label_generator=test_datagen.flow_from_dataframe(
  dataframe=testdf,
  directory="./urban/label_gen",
  x_col="slice_file_name",
  y_col="class",
  batch_size=2,
  seed=42,
  shuffle=False,
  class_mode="categorical",
  target_size=(64,64))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (label_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames

model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
# model_wo_sm = model

test_generator.reset()
print(len(predictions))

counter = 0
for batch in test_generator:
  # LRP epsilon
  analyzer = innvestigate.create_analyzer("lrp.epsilon",model_wo_sm)
  a = analyzer.analyze(batch)
  a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
  a /= np.max(np.abs(a))
  # LRP flat A
  analyzerFlat = innvestigate.create_analyzer("lrp.sequential_preset_a_flat",model_wo_sm)
  b = analyzerFlat.analyze(batch)
  b = b.sum(axis=np.argmax(np.asarray(b.shape) == 3))
  b /= np.max(np.abs(b))

  if counter*32+1 >= cat_size:
    break

  idx = (test_generator.batch_index - 1) * test_generator.batch_size
  for i in range(0,len(batch)):
    print(test_generator.filenames[idx : idx + test_generator.batch_size][i])

    # Save image
    plt.imshow(batch[i], cmap="seismic", clim=(-1, 1))
    plt.axis('off')
    filename = './images/'+ cat + '/' + remove_ext(filenames[counter*32+i]) + "_" + predictions[counter*32+i] + '.png'
    plt.savefig(filename)

    # Save LRPepsilon
    plt.imshow(a[i], cmap="seismic", clim=(-1, 1))
    plt.axis('off')
    filename = './images/' + cat + '/' + remove_ext(filenames[counter*32+i]) + '_lrp_epsilon.png'
    plt.savefig(filename)

    # Save LRPflat
    plt.imshow(b[i], cmap="seismic", clim=(-1, 1))
    plt.axis('off')
    filename = './images/' + cat + '/' + remove_ext(filenames[counter*32+i]) + '_lrp_flat.png'
    plt.savefig(filename)
    plot.close()
    plot.close('all')

  print("batch: " + str(test_generator.batch_index))
  counter += 1
