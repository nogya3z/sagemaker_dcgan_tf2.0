FROM tensorflow/tensorflow:nightly-custom-op-gpu-ubuntu16-cuda10.0


RUN pip install sagemaker-containers scipy imageio matplotlib numpy Pillow tensorflow-gpu

ENTRYPOINT ["python", "train"]
#ENV SAGEMAKER_PROGRAM dcgan.py

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY dcgan.py /opt/ml/code/dcgan.py
COPY read_tfrecords.py /opt/ml/code/read_tfrecords.py
COPY train /opt/ml/code/train

WORKDIR /opt/ml/code
