import keras
import tensorflow
import argparse
import os, gc

import DataLoader

docstring = "Train an instance of Inception-ResNet V2 for regression instead " \
            "of classification. Instead of classes, the targets are vectors " \
            "derived from GloVe vectors for the human-readable labels for " \
            "each class."

parser = argparse.ArgumentParser(description=docstring)
parser.add_argument('--load_dir',
                    help='Directory containing models to be tested.',
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

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True  # Claim GPU memory as needed
session = tensorflow.Session(config=config)

# Do some preprocessing to help with translating between classes and vectors
DataLoader.build_maps(label_file=args['labels'],
                      vector_file=args['vectors'],
                      class_to_synset_file=args['syn_map']
                      )

validation_data = DataLoader.get_val_loader(args['valid_dir'],
                                            batch_size=64)
decoder = DataLoader.Decoder()

for model_file in sorted(os.listdir(args['load'])):
  correct_count = 0

  if model_file[-3:] == '.h5':
    gc.collect()
    print("Trying model", model_file)
    try:
      model = keras.models.load_model(os.path.join(args['load'], model_file))
    except:
      raise RuntimeError('Failed to load model.')

    optim = keras.optimizers.Nadam()  # Needed to compile

    model.compile(optimizer=optim,
                  loss='mean_squared_error')
    for image_set in validation_data:
      predictions = model.predict(image_set[0])
      for image in range(image_set[0].shape[0]):  # For every image in input
        true_label = decoder.decode(image_set[1][image])
        pred_label = decoder.decode(predictions[image])
        if true_label == pred_label:
          correct_count += 1
    validation_data.on_epoch_end()
    del model   # Memory leak without this?
    print(model_file, ':', correct_count, '/', 64 * len(validation_data))
