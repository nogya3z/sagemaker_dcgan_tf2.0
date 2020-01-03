from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import horovod.tensorflow as hvd
import math
import numpy as np
import os
import tensorflow as tf
import time

from PIL import Image

from read_tfrecords import read
from tensorflow.keras import layers

BATCH_SIZE = 256
#horovod resources:
#https://github.com/horovod/horovod/blob/master/examples/tensorflow2_mnist.py
#https://github.com/horovod/horovod/blob/master/examples/keras_imagenet_resnet50.py#L166
hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')



#https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid

#https://github.com/skywall34/BabyGan/blob/master/DCGAN_CelebA.ipynb
#the structure for the generator and discriminator was integrated from the link above
def make_generator_model():
  model = tf.keras.Sequential()
  

  model.add(layers.Dense(4*4*1024, use_bias = False, input_shape = (100,)))
  

  model.add(layers.BatchNormalization())
  
 
  model.add(layers.LeakyReLU())
  
  
  model.add(layers.Reshape(( 4, 4, 1024)))

  assert model.output_shape == (None, 4, 4, 1024) # Note: None is the batch size
  

  model.add(layers.Conv2DTranspose(512, (5, 5), strides = (2,2), padding = "same", use_bias = False))
  assert model.output_shape == (None, 8, 8, 512)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())


  model.add(layers.Conv2DTranspose(256, (5,5), strides = (2,2), padding = "same", use_bias = False))
  assert model.output_shape == (None, 16, 16, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())


  model.add(layers.Conv2DTranspose(128, (5,5), strides = (2,2), padding = "same", use_bias = False))
  assert model.output_shape == (None, 32, 32, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  
  model.add(layers.Conv2DTranspose(3, (5,5), strides = (2,2), padding = "same", use_bias = False, activation = "tanh"))
  print(model.output_shape)
  assert model.output_shape == (None, 64, 64, 3)
  
  
  return model

generator = make_generator_model()

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) 
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
     
    return model


discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss   

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4* hvd.size())
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4* hvd.size())

#checkpoint_dir = model_dir
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images, epoch):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gen_tape = hvd.DistributedGradientTape(gen_tape)
  disc_tape = hvd.DistributedGradientTape(disc_tape)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  if epoch == 0:
    hvd.broadcast_variables(generator.variables, root_rank=0)
    hvd.broadcast_variables(discriminator.variables, root_rank=0)
    hvd.broadcast_variables(generator_optimizer.variables(), root_rank=0)
    hvd.broadcast_variables(discriminator_optimizer.variables(), root_rank=0)


    
def generate_and_save_images(model, epoch, test_input, model_dir):
  img_folder = model_dir+"/images"
  if not os.path.exists(img_folder):
    os.makedirs(img_folder)
  img_location = img_folder+ "/000image_at_epoch_{:04d}.png".format(epoch)
  predictions = model(test_input, training=False)
  predictions = predictions.numpy()
  predictions = (predictions*127.5)+127.5
  predictions = np.uint8(predictions)
  images = [Image.fromarray(prediction, 'RGB') for prediction in predictions]
  pil_grid(images, 4).save(img_location)
  #pil_grid(images, 4).show()

    
def train(data_dir, epochs, model_dir):
  print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
  print("Name of GPU's: ", (tf.config.experimental.list_physical_devices('GPU')))


  print(os.listdir(data_dir))
  files = os.listdir(data_dir)
  input_files = [os.path.join(data_dir, file) for file in files ]
  print(input_files)
  dataset = read(input_files)['inputs']
    
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
  checkpoint_prefix = model_dir + "/checkpoint"

  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch, epoch)

    # Produce images for the GIF as we go
    generate_and_save_images(generator,
                             epoch + 1,
                             seed, model_dir)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0 and  hvd.rank() == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           seed, model_dir)


def main(data_dir, epochs, model_dir):
    epochs = int(math.ceil(epochs / hvd.size()))
    #adjust the epochs for distributed training.
    train(data_dir, epochs, model_dir)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        '--data-dir',
        default='/opt/ml/input/data/training',
        type=str,
        help='The directory where the TF-RECORDS are stored. Default: /opt/ml/input/data/training. This '
             'directory corresponds to the SageMaker channel named \'training\', which was specified when creating '
             'our training job on SageMaker')

    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        '--model-dir',
        default='/opt/ml/model',
        type=str,
        help='The directory where the model and produced images will be stored. Default: /opt/ml/model. This directory should contain all '
             'final model artifacts as Amazon SageMaker copies all data within this directory as a single object in '
             'compressed tar format.')

    args_parser.add_argument(
        '--epochs',
        type=int,
        default = 1000, 
        help='Number of Epochs')
    args = args_parser.parse_args()
    main(**vars(args))