import tensorflow as tf
import os

batch_size = 256
INPUT_TENSOR_NAME = "inputs"

def parser(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string)
        })
    image = features['image']
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.compat.v2.math.subtract(image, 127.5)
    image = tf.math.divide(image, 127.5)
    
    return image

#best batch maker
def make_batch(data_dir, batch_size):
    dataset = tf.data.TFRecordDataset(data_dir)#.repeat()

    dataset = dataset.map(parser, num_parallel_calls=batch_size)

    min_queue_examples = int(45000 * 0.4)

    dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    dataset = dataset.batch(batch_size)

    return dataset

def read(data_dir):
    with tf.device('/cpu:0'):
        image_batch = make_batch(data_dir, batch_size)
        return {INPUT_TENSOR_NAME: image_batch}
