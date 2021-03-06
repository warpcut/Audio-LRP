# Audio-LRP


## Feature extraction

Spectrogram is a standard sound visualization tool, showing the distribution of energy in both time and frequency.

### Mel-spectrogram

Spectrogram with the Mel Scale as its y axis, Mel Scale is constructed such that sounds of equal distance from each other on the Mel Scale, also “sound” to humans as they are equal in distance from one another.

![Mel-Spectrogram example](https://github.com/warpcut/Audio-LRP/blob/master/mel_example.png)

### Constant-Q spectrogram

Spectrogram of the constant-q transform, it has geometrically spaced center frequencies, it also increases time resolution towards higher frequencies, as the human auditory system.

![Constant-Q-Spectrogram example](https://github.com/warpcut/Audio-LRP/blob/master/constant-q_example.png)

## Model training
  Model trained on the following net:
  ```shell
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_1 (Conv2D)            (None, 62, 62, 32)        896
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 60, 60, 64)        18496
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 30, 30, 64)        0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 30, 30, 64)        0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 28, 28, 64)        36928
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 26, 26, 64)        36928
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 64)        0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 13, 13, 64)        0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 11, 11, 128)       73856
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 9, 9, 128)         147584
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 5, 5, 128)         0
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 5, 5, 128)         0
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3200)              0
    _________________________________________________________________
    dense_512 (Dense)            (None, 512)               1638912
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 512)               0
    _________________________________________________________________
    dense_out (Dense)            (None, 10)                5130
    =================================================================
    Total params: 1,958,730
    Trainable params: 1,958,730
    Non-trainable params: 0
    _________________________________________________________________
  ```
## Layer relevance propagation

### LRP-epsilon

### LRP-flat

## How to use

1. Clone the repository
2. `cd Audio-LRP`
3. `docker build -t lrp-pytf .`
4. `docker run -it --rm lrp-pytf`
5. Once inside the new shell, run the scripts
6. To tranfer the data generated by the scripts, open a new terminal window

```shell
docker ps
```
CONTAINER-ID

9b81f107614b
```shell
docker cp <CONTAINER-ID>:./src/. ./dst
```

### Dataset
We are working with the [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.
The dataset has been pre-processed to make it uniform in duration and sampling frequency.

In this folder the feature extraction process has already been done, you can find the extracted mel-spectrogram inside urban/train, the feature are than reorganized into specific folders for the analysis part.
