# Overview

This proof-of-concept generates output vectors for each input image. The targets for regression are derived from GloVe vectors for the human-readable labels of each class. 

A model is available [here](https://drive.google.com/open?id=1bEBAp7XkrjPM6NjkqUXTR0TyPRdH-QcM) that was trained for 14 epochs; this cutoff was arbitrarily selected in order to free up the GPU for other work, so there's probably still some improvement in store. The model has a top-1 validation accuracy of about 60.3%.

## Why Is This Interesting?

This is a proof-of-concept linking NLP word embeddings and classic computer vision classification. This is something I'm hoping will be a step toward more robust computer vision, as we try to build networks that are better at handling novel inputs - for example, I'd like classifiers that can output something along the lines of "This is a dog, but not a breed I've seen before".

Of course, on its own, it's not very interesting - it's actually kind of a garbage classifier compared to the Inception-Resnet V2 network it was based on, and all it really establishes is that word embedding vectors are distinct enough that they can be distinguished from one another (which we already knew). On the other hand, it does have the advantage that its outputs are interpretable - there's an obvious metric comparing any given output to the known inputs the network was trained on.

It's probably also highly-related to image captioning work that's already been done. Need to read up on that, and would appreciate any suggestions.

# Training Results

Some sample results showing how classifications evolved through training for a few different images:


Epoch | The Good | The Bad | The Weird
----- | -------- | ------- | ---------
N/A | ![Red-Breasted Merganser Bird](/MiscImages/bird_small.png) | ![Submarine](/MiscImages/submarine_small.png) | ![American Lobster](/MiscImages/lobsters_small.png) 
0 | potter's wheel | potter's wheel | potter's wheel
1 | great egret bird | yellow lady's slipper orchid | yellow lady's slipper orchid
2 | red-breasted merganser bird | hand plane | yellow lady's slipper orchid
3 | black swan | potter's wheel | red king crab
4 | red-breasted merganser bird | assault rifle | red king crab
5 | red-breasted merganser bird | potter's wheel | American lobster
6 | red-breasted merganser bird | potter's wheel | cowboy boot
7 | red-breasted merganser bird | potter's wheel | American lobster
8 | red-breasted merganser bird | potter's wheel | American lobster
9 | red-breasted merganser bird | potter's wheel | cowboy boot
10 | red-breasted merganser bird | potter's wheel | cowboy boot
11 | red-breasted merganser bird | potter's wheel | potter's wheel
12 | red-breasted merganser bird | potter's wheel | cowboy boot
13 | red-breasted merganser bird | potter's wheel | American black bear
14 | red-breasted merganser bird | potter's wheel | American lobster

One thing worth noting is that "potter's wheel" is the closest label to an all-zero vector, which probably accounts for its frequency as a wrong answer, especially at the start. I don't have a good answer for "yellow lady's slipper orchid".

The validation accuracy over training epochs looks like this:

![Validation Accuracy](/MiscImages/Figure_1.png)

Note that only 49792 images were used in validating each epoch, and the indexes were randomized each epoch so each epoch got judged on a slightly different set. For a proof-of-concept hobby project this doesn't bother me.

# Details

`Main.py` runs the actual training process. All command line arguments are required except for a pretrained model to load. If none is provided, randomly-initialized weights are used.

The specific labels used to train the provided model are a slightly augmented variant of the labels in [imagenet-simple-labels](https://github.com/anishathalye/imagenet-simple-labels). In an extremely subjective process, additional words were added to labels that seemed like they could be clearer (for example, changing "brambling" to "brambling bird").

A decoder is included for mapping a vector to a class, but this isn't involved in training. Because this decoder is (relatively speaking) very slow, it is not used to assess accuracy during training - a separate script, `TestEpochModels.py` evaluates accuracy on the validation set.

When training starts, this script also selects 50 random validation images. At the start of each epoch, it will save each image with its current, decoded label so you can see how it's doing. The format is:

`[sample directory]/[epoch]/[image index]_[true label]:[predicted label].png`
