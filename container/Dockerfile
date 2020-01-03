FROM horovod/horovod:0.18.2-tf2.0.0-torch1.3.0-mxnet1.5.0-py3.6-gpu



RUN pip install sagemaker-containers scipy imageio matplotlib numpy Pillow



ENTRYPOINT ["python", "train"]


# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY dcgan.py /opt/ml/code/dcgan.py
COPY read_tfrecords.py /opt/ml/code/read_tfrecords.py
COPY train /opt/ml/code/train
COPY hvdmnist.py /opt/ml/code/hvdmnist.py

WORKDIR /opt/ml/code