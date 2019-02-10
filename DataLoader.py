import keras
import numpy
import random
import PIL
from PIL import ImageEnhance
import scipy.spatial
import json, os, re, pickle, sys

# Data structures for various mappings. We have 4 different values to keep
# track of for each conceptual "thing" -
#   * Its plain English label, useful for understanding what's going on
#   * Its synset string, useful for file paths
#   * Its representation vector, used as the network's training target
#   * Its class number, useful for decoding vectors to other values
# Filthy global variables may be factored out one day but for now are very
# convenient.

# Maps a synset to a word vector.
synset_to_vector = {}

# Maps a synset to a human-readable label.
synset_to_name = {}

# Maps a human-readable label to a synset.
name_to_synset = {}

# Maps each class number to its synset, after filtering out synsets that won't
# be used because their labels aren't represented in the word vectors.
class_to_synset = []


def insert_data(class_num, label, vector, base_class_to_synset):
  global synset_to_name, synset_to_vector, name_to_synset, class_to_synset
  # Map label to its synset
  name_to_synset[label] = base_class_to_synset[class_num]

  # Look up the synset by class number, map to label
  synset_to_name[base_class_to_synset[class_num]] = label

  # Map a synset to its vector
  synset_to_vector[name_to_synset[label]] = vector

  # Add the synset to the list of synsets we'll train with
  class_to_synset.append(name_to_synset[label])


def build_maps(label_file,
               vector_file,
               class_to_synset_file,
               pre_vectors='vecs.pkl',
               pre_synsets='syns.pkl',
               pre_labels='lbls.pkl'):
  global synset_to_name, synset_to_vector, name_to_synset, class_to_synset

  vectors = None

  # Maps each synset to its traditional ImageNet classifier class.
  synset_to_base_class = {}

  try:
    with open(pre_vectors, 'rb') as file:
      synset_to_vector = pickle.load(file)
    with open(pre_synsets, 'rb') as file:
      class_to_synset = pickle.load(file)
    with open(pre_labels, 'rb') as file:
      synset_to_name = pickle.load(file)
  except IOError:   # No pre-built label/vector maps, so we build one
    with open(label_file, "r") as file:
      labels = json.load(file)

    # Map between traditional ImageNet classes and synset strings
    base_class_to_synset = []
    with open(class_to_synset_file, "r") as file:
      for line in file.readlines():
        synset_to_base_class[line.split(' ')[0]] = len(base_class_to_synset)
        base_class_to_synset.append(line.split(' ')[0])

    # Construct dictionary of all known word representations
    vector_size = None
    with open(vector_file, "r") as f:
      vectors = {}
      for line in f.readlines():
        splitline = line.split(' ')
        vectors[splitline[0]] = numpy.array([float(x) for x in splitline[1:]])
        if vector_size is None:
          vector_size = vectors[splitline[0]].shape[0]

    # Check to see if each unmodified label appears in the dataset. If not, a
    # substitute is computed. First, stripping single quotes is tried; if that
    # fails, the label is split on spaces and hyphens, then the vector for the
    # label is the mean of the components. If a component isn't present, one
    # last try is made by stripping apostrophes from the component. If a
    # component still cannot be found, then the label is blacklisted and
    # training won't be performed with images from it.

    print("The following labels do not appear in the list of word vectors. "
          "They will be omitted from training.\n"
          "=================================================================")
    for i in range(len(labels)):
      label = labels[i]
      if label in vectors.keys():
        # No problems or complications
        insert_data(i, label, vectors[label], base_class_to_synset)
      else:
        # Try stripping out single-quotes.
        stripped = re.sub("[']", '', label)
        if stripped in vectors.keys():
          insert_data(i, label, vectors[stripped], base_class_to_synset)
        else:
          # Try splitting into components
          label_parts = re.split(r'[ -]+', label)
          mean_vector = numpy.zeros(vector_size)  # Should exist
          abort = False
          for part in label_parts:
            if part in vectors.keys():
              mean_vector += vectors[part]
            else:
              # Check components for apostrophes
              stripped_part = re.sub("[']", '', part)
              if stripped_part not in vectors.keys():
                # Give up
                print(label, ":", part, "/", stripped_part)
                abort = True
                break
          if not abort:
            mean_vector /= float(len(label_parts))
            insert_data(i, label, mean_vector, base_class_to_synset)
    print("=================================================================")
    with open('vecs.pkl', 'wb') as file:
      pickle.dump(synset_to_vector, file)
    with open('syns.pkl', 'wb') as file:
      pickle.dump(class_to_synset, file)
    with open('lbls.pkl', 'wb') as file:
      pickle.dump(synset_to_name, file)

  if vectors is not None:   # It's too big
    del vectors


class Decoder(object):
  def __init__(self):
    self.synset_to_vector = synset_to_vector
    self.class_to_synset = class_to_synset
    self.synset_to_name = synset_to_name
    anchor_array = numpy.array([synset_to_vector[anchor]
                                for anchor in class_to_synset])

    # Build a KDTree that stores each vector in the valid classes. Decoder will
    # query the tree to find the nearest vector to the network's output.
    self.tree = scipy.spatial.KDTree(anchor_array)

  def decode(self,
             vector):
    class_num = self.tree.query(vector)[1]  # TODO: Add "Unknown" when distance too big
    return self.synset_to_name[self.class_to_synset[class_num]]


class DataLoader(keras.utils.Sequence):
  # Dataloader based on
  # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
  # With added data augmentation

  def __init__(self,
               samples,
               targets,
               data_dir,
               batch_size=4,
               dim=(299, 299),
               n_channels=3,
               target_size=300,
               shuffle=True,
               max_rgb_shift=15,
               max_rgb_noise=15,
               max_contrast_var=0.2,
               max_brightness_var=0.2,
               mirror=True,
               max_rotate=30,
               ):

    self.dim = dim
    self.batch_size = batch_size
    self.targets = targets
    self.samples = samples
    self.n_channels = n_channels
    self.target_size = target_size
    self.data_dir = data_dir
    self.shuffle = shuffle
    self.max_rgb_shift = max_rgb_shift
    self.max_rgb_noise = max_rgb_noise
    self.max_contrast_var = max_contrast_var
    self.max_brightness_var = max_brightness_var
    self.mirror = mirror
    self.max_rotate = max_rotate

    self.on_epoch_end()

  def on_epoch_end(self):
    """Updates indexes after each epoch"""
    self.indexes = numpy.arange(len(self.samples))
    if self.shuffle:
      numpy.random.shuffle(self.indexes)

  def __data_generation(self, list_ids_temp):
    """Generates data containing batch_size samples"""
    image_batch = numpy.empty((self.batch_size, *self.dim, self.n_channels))
    target_batch = numpy.empty((self.batch_size, self.target_size))

    for i, ID in enumerate(list_ids_temp):
      # File directories match the filename's first 9 characters
      filedir = os.path.join(self.data_dir, ID[:9])
      filepath = os.path.join(filedir, ID)
      imgarr = keras.preprocessing.image.load_img(filepath)

      # Random rotation
      if self.max_rotate > 0:
        angle = random.randrange(-1 * self.max_rotate, self.max_rotate)
        imgarr = imgarr.rotate(angle)

      # Center crop, assuming channels last
      width, height = imgarr.size
      new_height = int(width * 0.875)
      new_width = int(height * 0.875)
      left = (width - new_width) // 2
      top = (height - new_height) // 2
      right = (width + new_width) // 2
      bottom = (height + new_height) // 2
      imgarr.crop((left, top, right, bottom))

      # Rescale
      imgarr = imgarr.resize((299, 299), resample=PIL.Image.BILINEAR)

      # Vary contrast
      if self.max_contrast_var is not 0.0:
        con_var = (random.random() - 0.5) * 2 * self.max_contrast_var
        imgarr = ImageEnhance.Contrast(imgarr).enhance(1.0 + con_var)

      # Vary brightness
      if self.max_brightness_var is not 0.0:
        bri_var = (random.random() - 0.5) * 2 * self.max_brightness_var
        imgarr = ImageEnhance.Brightness(imgarr).enhance(1.0 + bri_var)

      # Mirror the image
      if self.mirror is True and random.random() > 0.5:
        imgarr.transpose(PIL.Image.FLIP_LEFT_RIGHT)

      # Convert to array instead of image
      imgarr = keras.preprocessing.image.img_to_array(
        imgarr,
        data_format=keras.backend.image_data_format(),
        dtype=keras.backend.floatx()
      )

      # Add RGB color shift
      if self.max_rgb_shift > 0:
        shifts = (numpy.random.rand(3) - 0.5 * 2)
        shifts = numpy.rint(shifts * self.max_rgb_shift)
        shift_vec = numpy.empty([299, 299, 3])
        for p in range(299):
          for q in range(299):
            shift_vec[p][q] = shifts
        imgarr = imgarr + shift_vec

      # Add RGB noise
      if self.max_rgb_noise > 0:
        # Random, 0-centered noise
        noise_volume = random.random() * self.max_rgb_noise
        noise_vec = (numpy.random.rand(299, 299, 3) - 0.5) * 2 * noise_volume
        noise_vec = numpy.rint(noise_vec)
        imgarr = imgarr + noise_vec

      # Clamp all pixel channels to [0, 255]
      imgarr = numpy.clip(imgarr, 0, 255)

      # Compress to -1 to 1 range
      imgarr = imgarr / 127.5
      imgarr = imgarr - 1.0

      # Store data and class
      image_batch[i] = imgarr
      target_batch[i] = synset_to_vector[self.targets[ID]]
    return image_batch, target_batch

  def __len__(self):
    """Returns the number of batches per epoch"""
    return int(numpy.floor(len(self.samples) / self.batch_size))

  def __getitem__(self, index):
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    samples_temp = [self.samples[k] for k in indexes]

    # Generate data
    image_batch, target_batch = self.__data_generation(samples_temp)

    return image_batch, target_batch


def get_train_loader(train_dir, batch_size=16):
  data_dir = train_dir
  samples = []
  targets = {}
  for syn_name in class_to_synset:
    for sample in os.listdir(data_dir + syn_name):
      samples.append(sample)
      targets[sample] = syn_name
  return DataLoader(samples,
                    targets,
                    data_dir,
                    batch_size=batch_size
                    )


def get_val_loader(valid_dir, batch_size=4):
  data_dir = valid_dir
  samples = []
  targets = {}
  for syn_name in class_to_synset:
    for sample in os.listdir(data_dir + syn_name):
      samples.append(sample)
      targets[sample] = syn_name

  return DataLoader(samples,
                    targets,
                    data_dir,
                    batch_size=batch_size,
                    max_rgb_shift=0,
                    max_rgb_noise=0,
                    max_contrast_var=0.0,
                    max_brightness_var=0.0,
                    mirror=False,
                    max_rotate=0
                    )
