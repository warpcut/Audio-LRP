# Audio-LRP


## Feature extraction

## Model training

## Layer relevance propagation

### LRP-epsilon

### LRP-flat

## How to use

1. Clone the repository
2. `cd Audio-LRP`
3. `docker build -t lrp-pytf .`
4. `docker run -it lrp-pytf`

### Dataset
We are working with the [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.
The dataset has been pre-processed to make it uniform in duration and sampling frequency.

In this folder the feature extraction process has already been done, you can find the extracted mel-spectrogram inside urban/train, the feature are than reorganized into specific folders for the analysis part.
