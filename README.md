This proof-of-concept generates output vectors for each input image. The targets for regression are derived from GloVe vectors for the human-readable labels of each class. For convenience, a decoder is included for mapping a vector to a class, but this isn't involved in training. When training starts, it also selects 50 random validation images. At the start of each epoch, it will save each image with its current decoded label in the filename so you can see how it's doing. The format is:

[sample_dir]/[epoch]/[image_index]_[true_label]:[predicted_label].png

Performance doesn't seem to be great but does generate some interesting results.

All parameters are required, except for a pretrained model to load. If none is provided, pretrained ImageNet weights are downloaded.
