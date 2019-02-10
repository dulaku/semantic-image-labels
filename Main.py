import keras
import tensorflow
import numpy
import argparse
import os, gc

import DataLoader

docstring = "Train an instance of Inception-ResNet V2 for regression instead " \
            "of classification. Instead of classes, the targets are vectors " \
            "derived from GloVe vectors for the human-readable labels for " \
            "each class."

parser = argparse.ArgumentParser(description=docstring)
parser.add_argument('--load', help='Model file to load.')
parser.add_argument('--save_dir',
                    help='Directory to save models in at each epoch.',
                    required=True)
parser.add_argument('--sample_dir',
                    help='Directory to save sample images in before each '
                         'epoch.',
                    required=True)
parser.add_argument('--train_dir',
                    help='Directory containing ImageNet training images.',
                    required=True)
parser.add_argument('--valid_dir',
                    help='Directory containing ImageNet validation images. '
                         'Assumes the same directory structure as is typical '
                         'for training data - subdirectories for each class '
                         'with names matching the relevant synset.',
                    required=True)
parser.add_argument('--syn_map',
                    help='Text file mapping ImageNet class numbers to their '
                         'synsets. See https://github.com/HoldenCaulfieldRye/ca'
                         'ffe/blob/master/data/ilsvrc12/synset_words.txt for '
                         'an example that also happens to include human-'
                         'readable labels.',
                    required=True)
parser.add_argument('--labels',
                    help='JSON file containing human-readable labels for '
                         'ImageNet classes. See https://github.com/anishathalye'
                         '/imagenet-simple-labels for an example.',
                    required=True)
parser.add_argument('--vectors',
                    help='Text file containing word vectors. See https://nlp.st'
                         'anford.edu/projects/glove/ for the expected format.',
                    required=True)
args = vars(parser.parse_args())


class SampleImages(keras.callbacks.Callback):
  def __init__(self, num_samples, decoder):
    super(SampleImages, self).__init__()
    self.num_samples = num_samples
    self.decoder = decoder
    self.sample_images = []

  def on_train_begin(self, logs={}):
    sample_data = DataLoader.get_val_loader(args['valid_dir'],
                                            batch_size=self.num_samples)
    data = sample_data.__getitem__(0)   # Get the first batch from this loader.
    for image in range(data[0].shape[0]):   # For every image tensor in input
      true_label = self.decoder.decode(data[1][image])
      self.sample_images.append({"input": data[0][image],
                                 "true_label": true_label})

  def on_epoch_begin(self, epoch, logs={}):
    gc.collect()  # Shouldn't be needed, done just in case
    counter = 0   # Could iterate over range(len()) but this seems more readable
    for sample in self.sample_images:
      # Add a dimension to turn a single image into a "batch" of 1, then predict
      out_vector = self.model.predict(numpy.expand_dims(sample["input"], 0),
                                      batch_size=1)
      # Then take the only prediction vector in the "batch" of predictions
      label = self.decoder.decode(out_vector[0])
      path_base = os.path.join(args['sample_dir'], str(epoch))
      filename = str(counter) + "_" + sample["true_label"] + ":" + label \
                 + ".png"
      try:
        keras.preprocessing.image.save_img(os.path.join(path_base, filename),
                                           sample["input"])
      except IOError:
        os.mkdir(path_base)
        keras.preprocessing.image.save_img(os.path.join(path_base, filename),
                                           sample["input"])
      counter += 1


config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True  # Claim GPU memory as needed
session = tensorflow.Session(config=config)

# Do some preprocessing to help with translating between classes and vectors
DataLoader.build_maps(label_file=args['labels'],
                      vector_file=args['vectors'],
                      class_to_synset_file=args['syn_map']
                      )

train_data = DataLoader.get_train_loader(args['train_dir'])
validation_data = DataLoader.get_val_loader(args['valid_dir'])

try:
  base_model = keras.models.load_model(args['load'])
except:
  print("Backed up model unavailable, starting fresh...")
  base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights=None
  )

outlayer = base_model.output
outlayer = keras.layers.GlobalAveragePooling2D()(outlayer)
outlayer = keras.layers.Dense(300, activation='linear')(outlayer)

model = keras.Model(inputs=base_model.input,
                    outputs=outlayer)

optim = keras.optimizers.Nadam()
decode = DataLoader.Decoder()
sampler = SampleImages(50, decode)
checkpoint_format = os.path.join(args['save_dir'],
                                 'train_weights.{epoch:02d}-{val_loss:.2f}.h5')

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=2),
             keras.callbacks.ModelCheckpoint(filepath=checkpoint_format,
                                             monitor='val_loss',
                                             save_best_only=False),
             sampler,
             ]

model.compile(optimizer=optim,
              loss='mean_squared_error')

model.fit_generator(
  epochs=100,
  verbose=1,
  max_queue_size=20,
  workers=24,
  use_multiprocessing=True,
  generator=train_data,
  validation_data=validation_data,
  callbacks=callbacks,
)
