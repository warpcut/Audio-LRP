import warnings
warnings.simplefilter('ignore')
import imp
import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils
from keras import regularizers, optimizers
import pandas as pd
from keras.layers.advanced_activations import LeakyReLU
import pylab
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
traindf["slice_file_name"]=traindf["slice_file_name"].apply(append_ext)

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
