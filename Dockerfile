FROM tensorflow/tensorflow:1.15.2-py3

RUN apt-get update
RUN apt-get install -y git
RUN pip install pandas librosa matplotlib theano cntk pillow Keras
RUN pip install git+https://github.com/jaraco/path.git
RUN pip install innvestigate memory-profiler
RUN apt-get install -y ffmpeg
RUN pip install -q pyyaml h5py
RUN apt-get install -y vim

ADD urban /urban
ADD models /models
ADD images /images
ADD train.py /
ADD test_organizer.py /
ADD analyze.py /


# CMD [ "python", "./my_test.py" ]
